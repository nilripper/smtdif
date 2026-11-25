/* smtfeat: structural features of every formula fed to it */
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};

use smtdif::buf::Buf;
use smtdif::feat::Feat;
use smtdif::flags::Flags;
use smtdif::sexp::Sexp;
use smtdif::stream::stream;
use smtdif::tar::{istar, Tar};
use smtdif::{die, Res};

struct Out {
	w: BufWriter<io::StdoutLock<'static>>,
	/* -u: list the heads met in terms that the vocabulary does not know */
	long: bool,
	first: bool,
}

impl Out {
	fn
	emit(&mut self, f: &Feat) -> io::Result<()>
	{
		if(self.long){
			for (name, n) in &f.unknown{
				writeln!(self.w, "{}\t{}\t{}", f.id, String::from_utf8_lossy(name), n)?;
			}
			return Ok(());
		}
		if(self.first){
			writeln!(self.w, "{}", Feat::header())?;
			self.first = false;
		}
		writeln!(self.w, "{}", f.row())
	}
}

fn
scan<R: Read>(out: &mut Out, id: &str, logic: &str, in_: Buf<R>) -> Res<()>
{
	let mut sx = Sexp::new(in_);
	let mut f = Feat::new(id, logic);
	f.scan(&mut sx).map_err(|e| format!("{}: {}", id, e))?;
	f.bytes = sx.tell();
	out.emit(&f).map_err(|e| format!("{}: {}", id, e))
}

/* "non-incremental/QF_BV/family/x.smt2" is the benchmark QF_BV/family/x.smt2 of logic QF_BV */
fn
unnest(path: &str) -> (String, String)
{
	let p = path.strip_prefix("./").unwrap_or(path);
	let p = match p.split_once('/'){
		Some(("non-incremental", rest)) | Some(("incremental", rest)) => rest,
		_ => p,
	};
	match p.split_once('/'){
		Some((logic, rest)) => (logic.to_string(), format!("{}/{}", logic, rest)),
		None => (String::new(), p.to_string()),
	}
}

fn
feed<R: Read>(out: &mut Out, name: &str, r: R) -> Res<()>
{
	let r = io::BufReader::with_capacity(1 << 18, r);
	let st = stream(r).map_err(|e| format!("{}: {}", name, e))?;
	let mut b = Buf::new(st);
	if(istar(&mut b).map_err(|e| format!("{}: {}", name, e))?){
		let mut tar = Tar::new(b);
		loop {
			let path = match tar.entry().map_err(|e| format!("{}: {}", name, e))?{
				Some(p) => p.to_string(),
				None => return Ok(()),
			};
			let (logic, id) = unnest(&path);
			scan(out, &id, &logic, Buf::new(&mut tar))?;
		}
	}
	scan(out, name, "", b)
}

fn
run(fl: &mut Flags, out: &mut Out) -> Res<()>
{
	let files = fl.argv.clone();
	if(files.is_empty()){
		return feed(out, "-", io::stdin());
	}
	for f in &files{
		if(f == "-"){
			feed(out, f, io::stdin())?;
		}else{
			let file = File::open(f).map_err(|e| format!("can't open {}: {}", f, e))?;
			feed(out, f, file)?;
		}
	}
	Ok(())
}

fn
main()
{
	let mut fl = Flags::new();
	fl.parse("u");
	let name = fl.name().to_string();
	let mut out = Out { w: BufWriter::new(io::stdout().lock()), long: fl.has('u'), first: true };
	match run(&mut fl, &mut out).and_then(|_| out.w.flush().map_err(|e| e.to_string())){
		Ok(()) => {}
		Err(e) => die(&name, &e),
	}
}
