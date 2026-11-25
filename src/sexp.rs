/* sexp: tokens of the smt-lib surface syntax */
use std::io::{self, Read};
use crate::buf::Buf;

pub enum Tok<'a> {
	Open,
	Close,
	Atom(&'a [u8]),
	Eof,
}

/* bytes that end an atom */
const DELIM: [bool; 256] = {
	let mut d = [false; 256];
	d[b' ' as usize] = true;
	d[b'\t' as usize] = true;
	d[b'\r' as usize] = true;
	d[b'\n' as usize] = true;
	d[0x0b] = true;
	d[0x0c] = true;
	d[b'(' as usize] = true;
	d[b')' as usize] = true;
	d[b';' as usize] = true;
	d[b'|' as usize] = true;
	d[b'"' as usize] = true;
	d
};

pub struct Sexp<R> {
	in_: Buf<R>,
	scr: Vec<u8>,
}

impl<R: Read> Sexp<R> {
	pub fn
	new(in_: Buf<R>) -> Sexp<R>
	{
		Sexp { in_, scr: Vec::new() }
	}

	/* bytes consumed so far */
	pub fn
	tell(&self) -> u64
	{
		self.in_.tell()
	}

	pub fn
	tok(&mut self) -> io::Result<Tok<'_>>
	{
		loop {
			let Some(c) = self.in_.get()? else {
				return Ok(Tok::Eof);
			};
			match(c){
				b' ' | b'\t' | b'\r' | b'\n' | 0x0b | 0x0c => {}
				b';' => self.until(b'\n')?,
				b'(' => return Ok(Tok::Open),
				b')' => return Ok(Tok::Close),
				b'|' => {
					self.until(b'|')?;
					return Ok(Tok::Atom(b"|"));
				}
				b'"' => {
					self.string()?;
					return Ok(Tok::Atom(b"\""));
				}
				_ => return self.atom(c),
			}
		}
	}

	/* eat up to and including the next end byte, or the end of the stream */
	fn
	until(&mut self, end: u8) -> io::Result<()>
	{
		loop {
			let b = self.in_.buf()?;
			let hit = b.iter().position(|c| *c == end);
			let n = match hit {
				Some(n) => n + 1,
				None => b.len(),
			};
			if(n == 0){
				return Ok(());
			}
			self.in_.take(n);
			if(hit.is_some()){
				return Ok(());
			}
		}
	}

	/* eat a string literal, backslash escapes included */
	fn
	string(&mut self) -> io::Result<()>
	{
		let mut esc = false;
		loop {
			let Some(c) = self.in_.get()? else {
				return Ok(());
			};
			if(esc){
				esc = false;
			}else if(c == b'\\'){
				esc = true;
			}else if(c == b'"'){
				return Ok(());
			}
		}
	}

	fn
	atom(&mut self, first: u8) -> io::Result<Tok<'_>>
	{
		self.scr.clear();
		self.scr.push(first);
		loop {
			let (hit, m) = {
				let b = self.in_.buf()?;
				(b.iter().position(|c| DELIM[*c as usize]), b.len())
			};
			if(m == 0){
				return Ok(Tok::Atom(&self.scr));
			}
			let n = hit.unwrap_or(m);
			/* whole in the buffer, first byte included: hand it out where it lies */
			if(hit.is_some() && self.scr.len() == 1 && self.in_.at_hand()){
				self.in_.unget();
				return Ok(Tok::Atom(self.in_.take(n + 1)));
			}
			/* the atom runs on past what is at hand: keep the part seen */
			self.scr.extend_from_slice(self.in_.take(n));
			if(hit.is_some()){
				return Ok(Tok::Atom(&self.scr));
			}
		}
	}
}

#[cfg(test)]
mod test {
	use super::*;

	fn
	toks(s: &str) -> Vec<String>
	{
		let mut sx = Sexp::new(Buf::new(s.as_bytes()));
		let mut v = Vec::new();
		loop {
			let t = match sx.tok().unwrap() {
				Tok::Open => "(",
				Tok::Close => ")",
				Tok::Atom(a) => &String::from_utf8_lossy(a) as &str,
				Tok::Eof => break,
			};
			v.push(t.to_string());
		}
		v
	}

	#[test]
	fn tokens()
	{
		let s = "(assert ; a comment with ) and (\n (bvadd x_1 |a b| #x0f #b01 \"s \\\" t\" 4.5e3))";
		assert_eq!(toks(s),
			["(", "assert", "(", "bvadd", "x_1", "|", "#x0f", "#b01", "\"", "4.5e3", ")", ")"]);
	}

	#[test]
	fn atom_across_refills()
	{
		/* an atom longer than the whole buffer still comes out whole */
		let mut s = String::from("(");
		s.push_str(&"v".repeat(crate::buf::BUF * 3));
		s.push_str(" x)");
		let t = toks(&s);
		assert_eq!(t.len(), 4);
		assert_eq!(t[1].len(), crate::buf::BUF * 3);
		assert_eq!(t[2], "x");
	}

	#[test]
	fn stream_ends_midway()
	{
		assert_eq!(toks("(assert (bvadd x"), ["(", "assert", "(", "bvadd", "x"]);
		assert_eq!(toks("(set-info :source |ab\ncd|)(exit"), ["(", "set-info", ":source", "|", ")", "(", "exit"]);
	}
}
