/* tar: the regular files of a tar stream, one at a time */
use std::io::{self, Read};
use crate::buf::Buf;

pub struct Tar<R> {
	in_: Buf<R>,
	name: String,
	size: u64,
	left: u64,
}

/* does the stream open with a tar header: "ustar" at byte 257 */
pub fn
istar<R: Read>(b: &mut Buf<R>) -> io::Result<bool>
{
	if(!b.need(262)?){
		return Ok(false);
	}
	Ok(&b.buf()?[257..262] == b"ustar")
}

/* a size field: octal digits, or base 256 under a set top bit */
fn
octal(f: &[u8]) -> u64
{
	let mut n = 0;
	if(f[0] & 0x80 != 0){
		for c in &f[1..]{
			n = n << 8 | *c as u64;
		}
		return n;
	}
	for c in f{
		if !(b'0'..=b'7').contains(c){
			break;
		}
		n = n * 8 + (c - b'0') as u64;
	}
	n
}

fn
field(f: &[u8]) -> String
{
	let end = f.iter().position(|c| *c == 0).unwrap_or(f.len());
	String::from_utf8_lossy(&f[..end]).into_owned()
}

impl<R: Read> Tar<R> {
	pub fn
	new(in_: Buf<R>) -> Tar<R>
	{
		Tar { in_, name: String::new(), size: 0, left: 0 }
	}

	pub fn
	size(&self) -> u64
	{
		self.size
	}

	/* advance to the next regular file and name it; none when the archive is over */
	pub fn
	entry(&mut self) -> io::Result<Option<&str>>
	{
		let mut long = String::new();
		let mut pax = String::new();
		loop {
			/* what is left of the entry just passed, and its padding */
			let pad = (512 - self.size % 512) % 512;
			self.in_.skip(self.left + pad)?;
			if(!self.in_.need(512)?){
				return Ok(None);
			}
			let h = self.in_.take(512).to_vec();
			if(h.iter().all(|c| *c == 0)){
				return Ok(None);
			}
			self.size = octal(&h[124..136]);
			match(h[156]){
				b'L' => long = self.text(self.size)?,
				b'x' => {
					let (path, size) = self.pax(self.size)?;
					if(path.len() > 0){
						pax = path;
					}
					if(size > 0){
						self.size = size;
					}
				}
				b'0' | 0 | b'7' => {
					let name = field(&h[..100]);
					self.name = if(pax.len() > 0){
						pax
					}else if(long.len() > 0){
						long
					}else{
						let prefix = field(&h[345..500]);
						if(prefix.len() > 0){ format!("{}/{}", prefix, name) } else { name }
					};
					self.left = self.size;
					return Ok(Some(&self.name));
				}
				_ => {}
			}
			self.left = 0;
		}
	}

	fn
	text(&mut self, n: u64) -> io::Result<String>
	{
		self.in_.need(n as usize)?;
		let s = self.in_.take(n as usize).to_vec();
		let end = s.len() - usize::from(s.last() == Some(&0));
		Ok(String::from_utf8_lossy(&s[..end]).into_owned())
	}

	/* pax extended header records: "length key=value" lines; the path and size are kept */
	fn
	pax(&mut self, n: u64) -> io::Result<(String, u64)>
	{
		let s = self.text(n)?;
		let mut path = String::new();
		let mut size = 0;
		let mut rest = s.as_str();
		while let Some(sp) = rest.find(' '){
			let len = match rest[..sp].parse::<usize>(){
				Ok(l) if l > sp + 1 && l <= rest.len() => l,
				_ => break,
			};
			let rec = &rest[sp + 1..len];
			rest = &rest[len..];
			if let Some(eq) = rec.find('='){
				let val = rec[eq + 1..].trim_end_matches('\n');
				match(&rec[..eq]){
					"path" => path = val.to_string(),
					"size" => size = val.parse().unwrap_or(0),
					_ => {}
				}
			}
		}
		Ok((path, size))
	}
}

impl<R: Read> Read for Tar<R> {
	fn
	read(&mut self, dst: &mut [u8]) -> io::Result<usize>
	{
		if(self.left == 0){
			return Ok(0);
		}
		let b = self.in_.buf()?;
		let n = b.len().min(dst.len()).min(self.left as usize);
		dst[..n].copy_from_slice(self.in_.take(n));
		self.left -= n as u64;
		Ok(n)
	}
}

#[cfg(test)]
mod test {
	use super::*;
	use std::io::Read;

	/* one tar header block, with the fields a regular file needs */
	fn
	header(name: &str, typ: u8, size: usize) -> Vec<u8>
	{
		let mut h = vec![0u8; 512];
		h[..name.len()].copy_from_slice(name.as_bytes());
		h[100..107].copy_from_slice(b"0000644");
		h[108..115].copy_from_slice(b"0001750");
		h[116..123].copy_from_slice(b"0001750");
		let s = format!("{:011o}", size);
		h[124..124 + s.len()].copy_from_slice(s.as_bytes());
		h[148..156].copy_from_slice(b"        ");
		h[156] = typ;
		h[257..263].copy_from_slice(b"ustar ");
		h[263..265].copy_from_slice(b"00");
		let sum: u64 = h.iter().map(|c| *c as u64).sum();
		let c = format!("{:06o}", sum);
		h[148..148 + c.len()].copy_from_slice(c.as_bytes());
		h
	}

	fn
	pad(mut v: Vec<u8>) -> Vec<u8>
	{
		while(v.len() % 512 != 0){
			v.push(0);
		}
		v
	}

	/* a reader that hands out a few bytes at a time, as a pipe would */
	struct Drip<'a>(&'a [u8]);

	impl Read for Drip<'_> {
		fn
		read(&mut self, dst: &mut [u8]) -> io::Result<usize>
		{
			let n = dst.len().min(7).min(self.0.len());
			dst[..n].copy_from_slice(&self.0[..n]);
			self.0 = &self.0[n..];
			Ok(n)
		}
	}

	fn
	names(t: &[u8]) -> Vec<(String, String)>
	{
		let mut tar = Tar::new(Buf::new(Drip(t)));
		let mut v = Vec::new();
		loop {
			let name = match tar.entry().unwrap(){
				Some(n) => n.to_string(),
				None => break,
			};
			let mut s = String::new();
			tar.read_to_string(&mut s).unwrap();
			v.push((name, s));
		}
		v
	}

	#[test]
	fn entries()
	{
		let mut t = pad(header("d/", b'5', 0));
		t.extend(pad(header("a.smt2", b'0', 5)));
		t.extend(b"(exit".iter().copied().chain(std::iter::repeat(0).take(507)));
		/* a long name: the record comes before the file it names */
		t.extend(pad(header("././@LongLink", b'L', 8)));
		let nul = [0u8; 1];
		t.extend(b"bbb.smt2".iter().copied().chain(nul).chain(std::iter::repeat(0).take(503)));
		t.extend(pad(header("bbb.smt", b'0', 700)));
		t.extend(vec![b'x'; 700]);
		t.extend(vec![0; 1024]);
		t.extend(b"junk after the end".iter().copied());
		let n = names(&t);
		assert_eq!(n.len(), 2);
		assert_eq!(n[0], ("a.smt2".to_string(), "(exit".to_string()));
		assert_eq!(n[1].0, "bbb.smt2");
		assert_eq!(n[1].1.len(), 700);
	}

	#[test]
	fn pax_path()
	{
		let rec = "23 path=long/name.smt2\n";
		let mut t = pad(header("xhdr", b'x', rec.len()));
		t.extend(rec.as_bytes().iter().copied().chain(std::iter::repeat(0).take(512 - rec.len())));
		t.extend(pad(header("short", b'0', 2)));
		t.extend(b"ok".iter().copied().chain(std::iter::repeat(0).take(510)));
		assert_eq!(names(&t), [("long/name.smt2".to_string(), "ok".to_string())]);
		assert!(istar(&mut Buf::new(&t[..])).unwrap());
	}
}
