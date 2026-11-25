/* smtimp: what the features are worth to the learners, split by split. two tables:
 * imp.tsv, by permutation: the score a chosen setting loses on validation when a
 * feature is shuffled among the rows. with features that move together, as the counts
 * of a formula do, that costs every feature about as much, so comes
 * ablate.tsv: the score of a forest, and of a plane, taught over one group of features
 * alone, and over all the features but that group. the settings are those smtfit chose,
 * and the rows are dealt as its seed deals them */
use std::io::Read;
use std::path::PathBuf;
use std::thread;

use smtdif::data::Data;
use smtdif::exp::{self, Setup, SCHEMES, TASKS};
use smtdif::flags::Flags;
use smtdif::model::{families, Fam};
use smtdif::prep;
use smtdif::table::Table;
use smtdif::{die, input, Res};

/* a family at its chosen setting, over the rows of a split and a task */
struct Ask {
	name: &'static str,
	setting: Vec<f64>,
	scheme: usize,
	task: usize,
	train: Vec<usize>,
	val: Vec<usize>,
}

/* the permutation of every feature, or the ablation of one group of them */
struct Job {
	ask: usize,
	permute: bool,
	group: &'static str,
}

fn
run<R: Read>(fl: &mut Flags, input: R, dir: &PathBuf) -> Res<()>
{
	let read = |name: &str| -> Res<Table> {
		let file = std::fs::File::open(dir.join(name)).map_err(|e| format!("{}: {}", name, e))?;
		Table::read(file).map_err(|e| format!("{}: {}", name, e))
	};
	let (time, outcome) = (read("time.tsv")?, read("outcome.tsv")?);
	/* the setting smtfit chose for a family under a split and a task, as its fit takes it */
	let chosen = |fam: &Fam, scheme: usize, task: usize| -> Res<Vec<f64>> {
		let t = &TASKS[task];
		let table = if(t.classes){ &outcome } else { &time };
		let (cs, ct, cm, cp) = (table.col("scheme")?, table.col("task")?, table.col("model")?, table.col("params")?);
		let found = table.rows.iter()
			.find(|r| r[cs] == SCHEMES[scheme] && r[ct] == t.name && r[cm] == fam.name)
			.ok_or(format!("{}: nothing chosen for {} under {} and {}", dir.display(), fam.name, SCHEMES[scheme], t.name))?;
		let grid = if(t.classes){ &fam.gridc } else { &fam.gridr };
		grid.iter().find(|p| fam.show(p) == found[cp]).cloned()
			.ok_or(format!("{}: the setting {} of {} is not one of its grid", dir.display(), found[cp], fam.name))
	};
	let limit = fl.num('t', 1200.0)?;
	let d = Data::read(&Table::read(input).map_err(|e| format!("dataset: {}", e))?, limit)?;
	let threads = fl.int('j', std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))?.max(1);
	let setup = Setup { seed: fl.num('r', 2026.0)? as u64, threads: 1, places: 1, limit };
	let repeats = fl.int('n', 5)?;
	/* the forest and the boosting everywhere, and a plane beside the forest:
	 * ridge over the times, logistic over the outcome */
	let mut asks: Vec<Ask> = Vec::new();
	for scheme in 0..SCHEMES.len(){
		for task in 0..TASKS.len(){
			for fam in families(){
				let plane = TASKS[task].classes && fam.name == "logistic" || !TASKS[task].classes && fam.name == "ridge";
				if(fam.name != "forest" && fam.name != "boost" && !plane){
					continue;
				}
				let setting = chosen(&fam, scheme, task)?;
				let (train, val, _, _) = exp::share(&d, &setup, scheme, task);
				asks.push(Ask { name: fam.name, setting, scheme, task, train, val });
			}
		}
	}
	/* the jobs: a permutation table for the forest and the boosting; for the forest and
	 * the plane, every group of features alone and left out, and all of them together */
	let mut jobs: Vec<Job> = Vec::new();
	for (at, ask) in asks.iter().enumerate(){
		if(ask.name == "forest" || ask.name == "boost"){
			jobs.push(Job { ask: at, permute: true, group: "" });
		}
		if(ask.name == "forest" || ask.name == "ridge" || ask.name == "logistic"){
			for group in ["all", "size", "width", "depth", "product", "array", "operators"]{
				jobs.push(Job { ask: at, permute: false, group });
			}
		}
	}
	let (d, fams) = (&d, families());
	let (d, fams, asks) = (&d, &fams, &asks);
	/* every job in a thread, a run of jobs a thread, each writing tables of its own */
	let mut all: Vec<Vec<String>> = vec![Vec::new(); jobs.len()];
	let run = (jobs.len() + threads - 1) / threads;
	thread::scope(|scope| {
		for (tables, part) in all.chunks_mut(run).zip(jobs.chunks(run)){
			scope.spawn(move || {
				for (table, job) in tables.iter_mut().zip(part){
					*table = work(d, fams, &asks[job.ask], job, repeats);
				}
			});
		}
	});
	let (mut imp, mut ablate) = (String::from("scheme\ttask\tmodel\tfeature\timportance\n"),
		String::from("scheme\ttask\tmodel\tgroup\twith\twithout\n"));
	for (job, lines) in jobs.iter().zip(all.iter()){
		if(job.permute){
			imp.push_str(&lines.concat());
		}else{
			ablate.push_str(&lines.concat());
		}
	}
	std::fs::write(dir.join("imp.tsv"), imp).map_err(|e| format!("imp.tsv: {}", e))?;
	std::fs::write(dir.join("ablate.tsv"), ablate).map_err(|e| format!("ablate.tsv: {}", e))?;
	eprintln!("{} fits over {} settings, their tables in {}", jobs.len(), asks.len(), dir.display());
	Ok(())
}

/* the lines one job writes: a feature a line under permutation; under ablation, the
 * score over the group alone and the score over all but it, on a line */
fn
work(d: &Data, fams: &[Fam], ask: &Ask, job: &Job, repeats: usize) -> Vec<String>
{
	let f = fams.iter().position(|fam| fam.name == ask.name).unwrap_or(0);
	let head = format!("{}\t{}\t{}\t", SCHEMES[ask.scheme], TASKS[ask.task].name, ask.name);
	if(job.permute){
		return exp::importance(d, ask.task, &ask.train, &ask.val, f, &ask.setting, repeats).iter()
			.map(|(feature, lost)| format!("{}{}\t{}\n", head, feature, exp::num(*lost))).collect();
	}
	let kept = prep::keep(d, &ask.train);
	let own = exp::group(d, &kept, job.group);
	let (with, without) = if(job.group == "all"){
		(exp::worth(d, ask.task, &ask.train, &ask.val, f, &ask.setting, &kept), String::new())
	}else{
		let rest: Vec<usize> = kept.iter().copied().filter(|j| !own.contains(j)).collect();
		(exp::worth(d, ask.task, &ask.train, &ask.val, f, &ask.setting, &own),
			exp::num(exp::worth(d, ask.task, &ask.train, &ask.val, f, &ask.setting, &rest)))
	};
	vec![format!("{}{}\t{}\t{}\n", head, job.group, exp::num(with), without)]
}

fn
main()
{
	let mut fl = Flags::new();
	fl.parse("");
	let name = fl.name().to_string();
	let dir = PathBuf::from(fl.get('o').unwrap_or("out"));
	match input(&fl.argv).and_then(|input| run(&mut fl, input, &dir)){
		Ok(()) => {}
		Err(e) => die(&name, &e),
	}
}
