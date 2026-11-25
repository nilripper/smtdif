/* smtfit: every family under both splits and all three tasks, its tables into a directory */
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

use smtdif::data::Data;
use smtdif::exp::{self, Gate, Setup, SCHEMES, TASKS};
use smtdif::flags::Flags;
use smtdif::table::Table;
use smtdif::{die, input, Res};

fn
run<R: Read>(fl: &mut Flags, input: R) -> Res<()>
{
	/* the fits run side by side on every processor; a fifth of them, at most, may hold a square matrix */
	let threads = match fl.int('j', 0)?{
		0 => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
		n => n.max(1),
	};
	let setup = Setup {
		seed: fl.num('r', 2026.0)? as u64,
		threads,
		places: fl.int('m', (threads / 5).max(1))?,
		limit: fl.num('t', 1200.0)?,
	};
	let dir = PathBuf::from(fl.get('o').unwrap_or("out"));
	let t = Table::read(input).map_err(|e| format!("dataset: {}", e))?;
	let d = Data::read(&t, setup.limit)?;
	eprintln!("{}: {} instances, {} features, {} families; {} threads, {} square matrices at a time",
		fl.name(), d.n, d.d, d.families.len(), threads, setup.places);
	let gate = Gate::new(setup.places);
	let clock = Instant::now();
	let mut exps = Vec::new();
	for scheme in 0..SCHEMES.len(){
		for task in 0..TASKS.len(){
			eprintln!("{}: split by {}, task {}", fl.name(), SCHEMES[scheme], TASKS[task].name);
			let name = fl.name().to_string();
			exps.push(exp::one(&d, &setup, scheme, task, &gate, &mut |m| eprintln!("{}: {}", name, m)));
		}
	}
	eprintln!("{}: the tables", fl.name());
	exp::tables(&d, &exps, &dir)?;
	eprintln!("{}: tables in {} after {:.0} s", fl.name(), dir.display(), clock.elapsed().as_secs_f64());
	Ok(())
}

fn
main()
{
	let mut fl = Flags::new();
	fl.parse("");
	let name = fl.name().to_string();
	match input(&fl.argv).and_then(|input| run(&mut fl, input)){
		Ok(()) => {}
		Err(e) => die(&name, &e),
	}
}
