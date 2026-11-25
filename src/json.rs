/* json: read a json document one token at a time, without keeping any of it */
use std::io::Read;
use crate::buf::Buf;
use crate::Res;

#[derive(Debug, PartialEq)]
pub enum Tok {
	Obj,
	Endobj,
	Arr,
	Endarr,
	Str(String),
	Num(f64),
	True,
	False,
	Null,
	Eof,
}

/* the bytes a number is made of */
const NUM: [bool; 256] = {
	const PARTS: &[u8] = b"0123456789-+.eE";
	let mut n = [false; 256];
	let mut i = 0;
	while(i < PARTS.len()){
		n[PARTS[i] as usize] = true;
		i += 1;
	}
	n
};

pub struct Json<R> {
	in_: Buf<R>,
}

/* commas and colons are passed over: who reads the tokens knows where they fall */
impl<R: Read> Json<R> {
	pub fn
	new(in_: Buf<R>) -> Json<R>
	{
		Json { in_ }
	}

	pub fn
	tok(&mut self) -> Res<Tok>
	{
		let io = |e: std::io::Error| format!("json: {}", e);
		loop {
			let Some(c) = self.in_.get().map_err(io)? else {
				return Ok(Tok::Eof);
			};
			return match(c){
				b' ' | b'\t' | b'\r' | b'\n' | b',' | b':' => continue,
				b'{' => Ok(Tok::Obj),
				b'}' => Ok(Tok::Endobj),
				b'[' => Ok(Tok::Arr),
				b']' => Ok(Tok::Endarr),
				b'"' => Ok(Tok::Str(self.string_body()?)),
				b't' => {
					self.word(b"rue").map_err(io)?;
					Ok(Tok::True)
				}
				b'f' => {
					self.word(b"alse").map_err(io)?;
					Ok(Tok::False)
				}
				b'n' => {
					self.word(b"ull").map_err(io)?;
					Ok(Tok::Null)
				}
				_ => self.number(c).map_err(io),
			};
		}
	}

	/* the next token, which must be a string */
	pub fn
	string(&mut self) -> Res<String>
	{
		match(self.tok()?){
			Tok::Str(s) => Ok(s),
			_ => Err("json: string wanted".to_string()),
		}
	}

	/* the next token, which must be a number */
	pub fn
	num(&mut self) -> Res<f64>
	{
		match(self.tok()?){
			Tok::Num(n) => Ok(n),
			_ => Err("json: number wanted".to_string()),
		}
	}

	/* the next token, which must be t */
	pub fn
	expect(&mut self, t: Tok) -> Res<()>
	{
		let got = self.tok()?;
		if(got != t){
			return Err(format!("json: {:?} wanted, {:?} came", t, got));
		}
		Ok(())
	}

	/* nothing may follow but white space */
	pub fn
	end(&mut self) -> Res<()>
	{
		match(self.tok()?){
			Tok::Eof => Ok(()),
			t => Err(format!("json: end wanted, {:?} came", t)),
		}
	}

	/* pass over one value, whatever it holds */
	pub fn
	skip(&mut self) -> Res<()>
	{
		let mut depth = 0usize;
		loop {
			match(self.tok()?){
				Tok::Obj | Tok::Arr => depth += 1,
				Tok::Endobj | Tok::Endarr => {
					if(depth == 0){
						return Err("json: value wanted".to_string());
					}
					depth -= 1;
				}
				Tok::Eof => return Err("json: end of stream inside a value".to_string()),
				_ => {}
			}
			if(depth == 0){
				return Ok(());
			}
		}
	}

	fn
	word(&mut self, rest: &[u8]) -> std::io::Result<()>
	{
		for c in rest{
			match(self.in_.get()?){
				Some(d) if d == *c => {}
				_ => return Err(std::io::Error::other("json: bad literal")),
			}
		}
		Ok(())
	}

	fn
	string_body(&mut self) -> Res<String>
	{
		let mut s = Vec::new();
		let io = |e: std::io::Error| format!("json: {}", e);
		loop {
			let c = self.in_.get().map_err(io)?.ok_or("json: string left open")?;
			if(c == b'"'){
				break;
			}
			if(c != b'\\'){
				s.push(c);
				continue;
			}
			let e = self.in_.get().map_err(io)?.ok_or("json: string left open")?;
			match(e){
				b'"' => s.push(b'"'),
				b'\\' => s.push(b'\\'),
				b'/' => s.push(b'/'),
				b'b' => s.push(8),
				b'f' => s.push(12),
				b'n' => s.push(b'\n'),
				b'r' => s.push(b'\r'),
				b't' => s.push(b'\t'),
				b'u' => self.unicode(&mut s)?,
				_ => return Err("json: bad escape".to_string()),
			}
		}
		String::from_utf8(s).map_err(|_| "json: text is not utf-8".to_string())
	}

	fn
	unicode(&mut self, s: &mut Vec<u8>) -> Res<()>
	{
		let mut c = self.hex4()?;
		if((0xd800..0xdc00).contains(&c)){
			/* the low half of a surrogate pair follows, as another escape */
			let io = |e: std::io::Error| format!("json: {}", e);
			if(self.in_.get().map_err(io)? != Some(b'\\') || self.in_.get().map_err(io)? != Some(b'u')){
				return Err("json: lone surrogate".to_string());
			}
			let d = self.hex4()?;
			if !(0xdc00..0xe000).contains(&d){
				return Err("json: lone surrogate".to_string());
			}
			c = 0x10000 + ((c - 0xd800) << 10) + (d - 0xdc00);
		}
		let c = char::from_u32(c).filter(|c| *c != '\0').ok_or("json: bad escape")?;
		let mut b = [0u8; 4];
		s.extend_from_slice(c.encode_utf8(&mut b).as_bytes());
		Ok(())
	}

	fn
	hex4(&mut self) -> Res<u32>
	{
		let mut n = 0;
		for _ in 0..4{
			let c = self.in_.get().map_err(|e| format!("json: {}", e))?.ok_or("json: bad escape")?;
			let d = match(c){
				b'0'..=b'9' => c - b'0',
				b'a'..=b'f' => c - b'a' + 10,
				b'A'..=b'F' => c - b'A' + 10,
				_ => return Err("json: bad escape".to_string()),
			};
			n = n << 4 | d as u32;
		}
		Ok(n)
	}

	fn
	number(&mut self, first: u8) -> std::io::Result<Tok>
	{
		let mut s = vec![first];
		loop {
			let (hit, m) = {
				let b = self.in_.buf()?;
				(b.iter().position(|c| !NUM[*c as usize]), b.len())
			};
			if(hit.is_some() || m == 0){
				s.extend_from_slice(self.in_.take(hit.unwrap_or(m)));
				break;
			}
			s.extend_from_slice(self.in_.take(m));
		}
		let t = std::str::from_utf8(&s).map_err(|_| std::io::Error::other("json: bad number"))?;
		t.parse::<f64>().map(Tok::Num).map_err(|_| std::io::Error::other("json: bad number"))
	}
}

#[cfg(test)]
mod test {
	use super::*;

	/* the tokens of s from the start, or from where j stands */
	fn
	rest(j: &mut Json<&[u8]>) -> Vec<Tok>
	{
		let mut v = Vec::new();
		loop {
			let t = j.tok().unwrap();
			let end = t == Tok::Eof;
			v.push(t);
			if(end){
				return v;
			}
		}
	}

	fn
	toks(s: &str) -> Vec<Tok>
	{
		rest(&mut Json::new(Buf::new(s.as_bytes())))
	}

	fn
	skip(s: &str) -> Vec<Tok>
	{
		let mut j = Json::new(Buf::new(s.as_bytes()));
		j.skip().unwrap();
		rest(&mut j)
	}

	fn
	s(s: &str) -> Tok
	{
		Tok::Str(s.to_string())
	}

	#[test]
	fn all_tokens()
	{
		assert_eq!(toks(r#" {"a": [1, -2.5e3, true, false, null], "b": {}} "#), [
			Tok::Obj,
			s("a"),
			Tok::Arr,
			Tok::Num(1.0),
			Tok::Num(-2500.0),
			Tok::True,
			Tok::False,
			Tok::Null,
			Tok::Endarr,
			s("b"),
			Tok::Obj,
			Tok::Endobj,
			Tok::Endobj,
			Tok::Eof,
		]);
	}

	#[test]
	fn escapes()
	{
		/* the json string with a quote, a backslash, a slash, a control character,
		 * a line feed, a two byte letter, a surrogate pair and a tab escaped in it;
		 * each of its backslashes is written b below, to keep this text plain */
		let b = 92u8 as char;
		let text = format!(r#""a{b}"b{b}{b}c{b}/dA{b}u0014B{b}nC{b}u00e9{b}ud83d{b}ude00{b}t""#);
		assert_eq!(toks(&text), [
			s("a\"b\\c/dA\u{14}B\nC\u{e9}\u{1f600}\t"),
			Tok::Eof,
		]);
		assert_eq!(toks(r#""x y""#), [s("x y"), Tok::Eof]);
	}

	#[test]
	fn skipping()
	{
		assert_eq!(skip(r#"[1, {"x": [[null]]}, "s"] true"#), [Tok::True, Tok::Eof]);
		assert_eq!(skip(r#"  "a\"b" 7"#), [Tok::Num(7.0), Tok::Eof]);
		assert_eq!(skip(r#"  {"a": {"b": [1, 2]}} null"#), [Tok::Null, Tok::Eof]);
	}

	#[test]
	fn stream_cut_short()
	{
		assert_eq!(toks(r#"{"a": [1, 2"#), [Tok::Obj, s("a"), Tok::Arr, Tok::Num(1.0), Tok::Num(2.0), Tok::Eof]);
		let mut j = Json::new(Buf::new(&br#""abc"#[..]));
		assert!(j.tok().is_err());
	}

	#[test]
	fn helpers()
	{
		let mut j = Json::new(Buf::new(&br#"{"k": "v", "n": 1.5}  "#[..]));
		assert_eq!(j.tok().unwrap(), Tok::Obj);
		assert_eq!(j.string().unwrap(), "k");
		assert_eq!(j.string().unwrap(), "v");
		assert_eq!(j.string().unwrap(), "n");
		assert_eq!(j.num().unwrap(), 1.5);
		assert_eq!(j.expect(Tok::Endobj), Ok(()));
		assert_eq!(j.end(), Ok(()));
		assert!(j.string().is_err());
	}
}
