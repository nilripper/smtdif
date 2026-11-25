/* buf: bytes from a reader, a buffer at a time */
use std::io::{self, ErrorKind, Read};

pub const BUF: usize = 1 << 16;

pub struct Buf<R> {
	r: R,
	b: Vec<u8>,
	pos: usize,
	len: usize,
	eof: bool,
	total: u64,
}

impl<R: Read> Buf<R> {
	pub fn
	new(r: R) -> Buf<R>
	{
		Buf { r, b: vec![0; BUF], pos: 0, len: 0, eof: false, total: 0 }
	}

	/* bytes consumed so far */
	pub fn
	tell(&self) -> u64
	{
		self.total
	}

	fn
	fill(&mut self) -> io::Result<bool>
	{
		self.pos = 0;
		self.len = 0;
		while(!self.eof){
			match(self.r.read(&mut self.b)){
				Ok(0) => self.eof = true,
				Ok(n) => {
					self.len = n;
					return Ok(true);
				}
				Err(e) if e.kind() == ErrorKind::Interrupted => {}
				Err(e) => return Err(e),
			}
		}
		Ok(false)
	}

	/* the bytes at hand; empty only at the end of the stream */
	pub fn
	buf(&mut self) -> io::Result<&[u8]>
	{
		if(self.pos == self.len && !self.eof){
			self.fill()?;
		}
		Ok(&self.b[self.pos..self.len])
	}

	/* n bytes from the head of buf(), consumed; n must not exceed their number */
	pub fn
	take(&mut self, n: usize) -> &[u8]
	{
		let s = &self.b[self.pos..self.pos + n];
		self.pos += n;
		self.total += n as u64;
		s
	}

	/* the next byte; none at the end of the stream */
	#[inline]
	pub fn
	get(&mut self) -> io::Result<Option<u8>>
	{
		if(self.pos == self.len && !self.eof){
			self.fill()?;
		}
		if(self.pos == self.len){
			return Ok(None);
		}
		let c = self.b[self.pos];
		self.pos += 1;
		self.total += 1;
		Ok(Some(c))
	}

	/* put back the byte get() just gave, if at_hand() says it is still there */
	#[inline]
	pub fn
	unget(&mut self)
	{
		self.pos -= 1;
		self.total -= 1;
	}

	/* is the byte get() just gave still in the buffer, where unget() reaches it */
	#[inline]
	pub fn
	at_hand(&self) -> bool
	{
		self.pos > 0
	}

	/* see to it that buf() offers n bytes, if the stream has them; tells whether it has */
	pub fn
	need(&mut self, n: usize) -> io::Result<bool>
	{
		if(n > self.b.len()){
			return Err(io::Error::new(ErrorKind::InvalidInput, "want more than a buffer full"));
		}
		while(self.len - self.pos < n && !self.eof){
			if(self.pos > 0){
				self.b.copy_within(self.pos..self.len, 0);
				self.len -= self.pos;
				self.pos = 0;
			}
			match(self.r.read(&mut self.b[self.len..])){
				Ok(0) => self.eof = true,
				Ok(m) => self.len += m,
				Err(e) if e.kind() == ErrorKind::Interrupted => {}
				Err(e) => return Err(e),
			}
		}
		Ok(self.len - self.pos >= n)
	}

	/* drop the next n bytes */
	pub fn
	skip(&mut self, mut n: u64) -> io::Result<()>
	{
		while(n > 0){
			let m = self.buf()?.len().min(n as usize);
			if(m == 0){
				return Err(io::Error::new(ErrorKind::UnexpectedEof, "truncated stream"));
			}
			self.take(m);
			n -= m as u64;
		}
		Ok(())
	}
}
