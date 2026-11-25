/* stream: undo zstd and gzip, as the first bytes say */
use std::io::{self, Read};

pub struct Stream<R: Read>(Dec<R>);

enum Dec<R: Read> {
	Zstd(ruzstd::decoding::StreamingDecoder<Back<R>, ruzstd::decoding::FrameDecoder>),
	Gz(flate2::read::MultiGzDecoder<Back<R>>),
	Raw(Back<R>),
}

/* the bytes read to tell the formats apart, put back in front of the rest */
struct Back<R> {
	r: R,
	head: [u8; 4],
	n: usize,
	pos: usize,
}

impl<R: Read> Read for Back<R> {
	fn
	read(&mut self, dst: &mut [u8]) -> io::Result<usize>
	{
		if(self.pos < self.n){
			let m = dst.len().min(self.n - self.pos);
			dst[..m].copy_from_slice(&self.head[self.pos..self.pos + m]);
			self.pos += m;
			return Ok(m);
		}
		self.r.read(dst)
	}
}

/* r with its compression, if any, taken off */
pub fn
stream<R: Read>(r: R) -> io::Result<Stream<R>>
{
	let mut back = Back { r, head: [0; 4], n: 0, pos: 0 };
	while(back.n < back.head.len()){
		let n = back.n;
		match(back.r.read(&mut back.head[n..])){
			Ok(0) => break,
			Ok(m) => back.n += m,
			Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
			Err(e) => return Err(e),
		}
	}
	let dec = if(back.head == [0x28, 0xb5, 0x2f, 0xfd]){
		Dec::Zstd(ruzstd::decoding::StreamingDecoder::new(back).map_err(|e| io::Error::other(e.to_string()))?)
	}else if(back.head[..2] == [0x1f, 0x8b]){
		Dec::Gz(flate2::read::MultiGzDecoder::new(back))
	}else{
		Dec::Raw(back)
	};
	Ok(Stream(dec))
}

impl<R: Read> Read for Stream<R> {
	fn
	read(&mut self, dst: &mut [u8]) -> io::Result<usize>
	{
		match(&mut self.0){
			Dec::Zstd(r) => r.read(dst),
			Dec::Gz(r) => r.read(dst),
			Dec::Raw(r) => r.read(dst),
		}
	}
}
