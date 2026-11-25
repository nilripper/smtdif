/* exp: the experiments. each task under a split by family and a split at random,
 * each family with its setting chosen by cross validation over the training rows */
use std::cmp::Reverse;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use crate::data::{Data, CLASSES};
use crate::model::{families, Fam, Mat, Model};
use crate::prep;
use crate::rng::Rand;
use crate::split::deal;
use crate::stat::{mae, r2, rmse, spearman, Conf};

pub static SCHEMES: [&str; 2] = ["family", "random"];

/* a task: what is predicted, and over which instances */
pub struct Task {
	pub name: &'static str,
	pub classes: bool,
	/* only the instances the solver finished */
	pub solved: bool,
}

pub static TASKS: [Task; 3] = [
	Task { name: "par2", classes: false, solved: false },
	Task { name: "solved", classes: false, solved: true },
	Task { name: "outcome", classes: true, solved: false },
];

/* the shares of training, validation and test; the folds of the cross validation */
pub static SHARE: [f64; 3] = [0.7, 0.15, 0.15];
pub static FOLDS: usize = 5;

/* how the fits are run: a seed for all that looks like chance, how many at once,
 * and how many of those may hold a square matrix over their training rows */
pub struct Setup {
	pub seed: u64,
	pub threads: usize,
	pub places: usize,
	pub limit: f64,
}

/* how far predictions fall from the truth */
#[derive(Clone, Default)]
pub struct Score {
	pub mae: f64,
	pub rmse: f64,
	pub rho: f64,
	pub r2: f64,
	pub acc: f64,
	/* the mean f1 over the classes */
	pub f1: f64,
	/* the f1 of the timeouts */
	pub f1to: f64,
	pub conf: Conf,
}

fn
confusion(y: &[f64], p: &[f64]) -> Conf
{
	let mut c = Conf::new(CLASSES.len());
	for (a, b) in y.iter().zip(p){
		c.add(*a as usize, *b as usize);
	}
	c
}

/* the one number a setting is chosen by: the error of the time, or the mean f1 */
pub fn
judge(classes: bool, y: &[f64], p: &[f64]) -> f64
{
	if(classes){ confusion(y, p).macro_f1() } else { mae(y, p) }
}

fn
score(classes: bool, y: &[f64], p: &[f64]) -> Score
{
	if(classes){
		let conf = confusion(y, p);
		return Score { acc: conf.accuracy(), f1: conf.macro_f1(), f1to: conf.f1(2), conf, ..Default::default() };
	}
	Score { mae: mae(y, p), rmse: rmse(y, p), rho: spearman(y, p), r2: r2(y, p), ..Default::default() }
}

/* the truth of rows under a task */
pub fn
truth(d: &Data, t: &Task, rows: &[usize]) -> Vec<f64>
{
	rows.iter().map(|i| if(t.classes){ d.c[*i] as f64 } else { d.y[*i] }).collect()
}

/* a family at one of its settings, taught by some rows and asked about others */
fn
fitrows(d: &Data, t: &Task, train: &[usize], probe: &[usize], fam: &Fam, p: &[f64]) -> Vec<f64>
{
	let kept = prep::keep(d, train);
	let mut a = prep::mat(d, train, &kept, t.classes);
	let mut b = prep::mat(d, probe, &kept, t.classes);
	prep::scale(&mut a, &mut b);
	let model = (fam.fit)(p, &a);
	ask(&*model, &b)
}

/* what a model says of every row of a matrix */
pub fn
ask(model: &dyn Model, m: &Mat) -> Vec<f64>
{
	(0..m.n).map(|i| model.predict(m.row(i))).collect()
}

/* how many fits may hold a square matrix at once: a count, waited on by sleeping */
pub struct Gate(AtomicUsize);

impl Gate {
	pub fn
	new(places: usize) -> Gate
	{
		Gate(AtomicUsize::new(places.max(1)))
	}

	/* a place, waited for */
	pub fn
	enter(&self) -> Place<'_>
	{
		loop {
			match(self.take()){
				Some(place) => return place,
				None => thread::sleep(Duration::from_millis(20)),
			}
		}
	}

	/* a place, when one is free */
	pub fn
	try_enter(&self) -> Option<Place<'_>>
	{
		self.take()
	}

	fn
	take(&self) -> Option<Place<'_>>
	{
		self.0.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| (n > 0).then(|| n - 1)).ok().map(|_| Place(self))
	}
}

/* a place at the gate, given back when the fit that holds it is over, or comes apart */
pub struct Place<'a>(&'a Gate);

impl Drop for Place<'_> {
	fn
	drop(&mut self)
	{
		(self.0).0.fetch_add(1, Ordering::SeqCst);
	}
}

/* one way of splitting under one task: the parts, and what every family came to */
pub struct Exp {
	pub scheme: usize,
	pub task: usize,
	pub train: Vec<usize>,
	pub val: Vec<usize>,
	pub test: Vec<usize>,
	pub folds: Vec<Vec<usize>>,
	/* for each family: its settings' scores under cross validation, the setting chosen,
	 * and its predictions on validation and on test */
	pub cv: Vec<Vec<f64>>,
	pub choice: Vec<usize>,
	pub valpred: Vec<Vec<f64>>,
	pub testpred: Vec<Vec<f64>>,
}

impl Exp {
	/* the family's setting as text */
	pub fn
	shown(&self, f: usize, fam: &Fam) -> String
	{
		let grid = if(TASKS[self.task].classes){ &fam.gridc } else { &fam.gridr };
		fam.show(&grid[self.choice[f]])
	}

	pub fn
	valscore(&self, d: &Data, f: usize) -> Score
	{
		score(TASKS[self.task].classes, &truth(d, &TASKS[self.task], &self.val), &self.valpred[f])
	}

	pub fn
	testscore(&self, d: &Data, f: usize) -> Score
	{
		score(TASKS[self.task].classes, &truth(d, &TASKS[self.task], &self.test), &self.testpred[f])
	}
}

struct Job {
	fam: usize,
	cfg: usize,
	part: usize,
}

/* how a seed deals the rows of a task: whole families to a part under the first
 * split, single rows under the second. the folds of the cross validation come
 * out of the training rows the same way. the order is training, validation, test, folds */
pub fn
share(d: &Data, setup: &Setup, scheme: usize, task: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<Vec<usize>>)
{
	let t = &TASKS[task];
	let world: Vec<usize> = (0..d.n).filter(|i| !t.solved || d.solved[*i]).collect();
	let whole = scheme == 0;
	let groups = |rows: &[usize]| whole.then(|| rows.iter().map(|i| d.group[*i]).collect::<Vec<usize>>());
	let mut rand = Rand::new(setup.seed + 10 * scheme as u64 + task as u64);
	let parts = deal(groups(&world).as_deref(), world.len(), &SHARE, &mut rand);
	let at = |p: usize| parts[p].iter().map(|i| world[*i]).collect::<Vec<usize>>();
	let (train, val, test) = (at(0), at(1), at(2));
	let fifths = vec![1.0 / FOLDS as f64; FOLDS];
	let folds: Vec<Vec<usize>> = deal(groups(&train).as_deref(), train.len(), &fifths, &mut rand)
		.iter().map(|f| f.iter().map(|i| train[*i]).collect()).collect();
	(train, val, test, folds)
}

/* the experiment of one scheme under one task */
pub fn
one(d: &Data, setup: &Setup, scheme: usize, task: usize, gate: &Gate, tell: &mut dyn FnMut(String)) -> Exp
{
	let t = &TASKS[task];
	let (train, val, test, folds) = share(d, setup, scheme, task);

	/* the training rows and the tried rows of each run: a fold of the cross validation,
	 * then the validation part, then the test part */
	let mut cuts: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
	for fold in &folds{
		let mut held = vec![false; d.n];
		for i in fold{
			held[*i] = true;
		}
		cuts.push((train.iter().copied().filter(|i| !held[*i]).collect(), fold.clone()));
	}
	cuts.push((train.clone(), val.clone()));
	cuts.push((train.clone(), test.clone()));

	let fams = families();
	let fams = &fams;
	let grid = |f: usize| if(t.classes){ &fams[f].gridc } else { &fams[f].gridr };
	let mut jobs: Vec<Job> = Vec::new();
	for f in 0..fams.len(){
		for cfg in 0..grid(f).len(){
			for part in 0..cuts.len(){
				jobs.push(Job { fam: f, cfg, part });
			}
		}
	}
	/* the dear families first, that all the threads have work to the end; those that
	 * hold a square matrix queue apart, that the others never wait on their places */
	let mut order: Vec<usize> = (0..jobs.len()).collect();
	order.sort_by_key(|j| Reverse(jobs[*j].fam));
	let mut plain: Vec<usize> = Vec::new();
	let mut square: Vec<usize> = Vec::new();
	for j in order{
		if(fams[jobs[j].fam].square){ square.push(j) } else { plain.push(j) }
	}
	/* the threads share all of this by reference; each owns a copy of the sending end */
	let (next_plain, next_square) = (AtomicUsize::new(0), AtomicUsize::new(0));
	let (plain, square, cuts, jobs) = (&plain, &square, &cuts, &jobs);
	let (next_plain, next_square) = (&next_plain, &next_square);
	let (done, seen) = mpsc::channel();
	let clock = Instant::now();
	let out: Vec<Vec<f64>> = thread::scope(move |scope| {
		for _ in 0..setup.threads{
			let done = done.clone();
			scope.spawn(move || {
				'work: loop {
					/* a square matrix goes when a place is free, that the places stay busy;
					 * a plain fit goes at once; and with no plain fit left, a place is worth the wait */
					let (at, place) = 'pick: loop {
						if let Some(place) = gate.try_enter(){
							let k = next_square.fetch_add(1, Ordering::SeqCst);
							if(k < square.len()){
								break 'pick (square[k], Some(place));
							}
						}
						let k = next_plain.fetch_add(1, Ordering::SeqCst);
						if(k < plain.len()){
							break 'pick (plain[k], None);
						}
						let k = next_square.fetch_add(1, Ordering::SeqCst);
						if(k < square.len()){
							break 'pick (square[k], Some(gate.enter()));
						}
						break 'work;
					};
					let job = &jobs[at];
					let p = grid(job.fam);
					let (train, probe) = &cuts[job.part];
					let pred = fitrows(d, t, train, probe, &fams[job.fam], &p[job.cfg]);
					drop(place);
					if(done.send((at, pred)).is_err()){
						break;
					}
				}
			});
		}
		/* the workers hold copies; with this one gone, the channel closes when they are done */
		drop(done);
		let mut out: Vec<Vec<f64>> = vec![Vec::new(); jobs.len()];
		let mut got = 0;
		while let Ok((j, pred)) = seen.recv(){
			out[j] = pred;
			got += 1;
			if(got % 200 == 0 || got == jobs.len()){
				tell(format!("{}/{}: {} fits of {} in {:.0} s", SCHEMES[scheme], t.name, got, jobs.len(), clock.elapsed().as_secs_f64()));
			}
		}
		if(got < jobs.len()){
			panic!("{} of {} fits came back", got, jobs.len());
		}
		out
	});
	let at = |f: usize, cfg: usize, part: usize| jobs.iter().position(|j| j.fam == f && j.cfg == cfg && j.part == part).unwrap_or(0);

	let mut e = Exp {
		scheme,
		task,
		train,
		val,
		test,
		folds: folds.clone(),
		cv: Vec::new(),
		choice: Vec::new(),
		valpred: Vec::new(),
		testpred: Vec::new(),
	};
	for f in 0..fams.len(){
		let mut cv = Vec::new();
		for cfg in 0..grid(f).len(){
			let (mut pooled, mut want) = (Vec::new(), Vec::new());
			for part in 0..FOLDS{
				want.extend(truth(d, t, &folds[part]));
				pooled.extend(out[at(f, cfg, part)].clone());
			}
			cv.push(judge(t.classes, &want, &pooled));
		}
		/* the best setting; the first of those as good, which is the plainest */
		let pick = |best: f64, s: f64| if(t.classes){ s > best } else { s < best };
		let mut choice = 0;
		for cfg in 1..cv.len(){
			if(pick(cv[choice], cv[cfg])){
				choice = cfg;
			}
		}
		let (valpred, testpred) = if(cv.is_empty()){
			(Vec::new(), Vec::new())
		}else{
			(out[at(f, choice, FOLDS)].clone(), out[at(f, choice, FOLDS + 1)].clone())
		};
		e.cv.push(cv);
		e.choice.push(choice);
		e.valpred.push(valpred);
		e.testpred.push(testpred);
	}
	e
}

/* the features by what they tell of a formula. a feature belongs to one group only, and
 * the counts of the operators belong to none of these: they are the rest */
pub static GROUPS: [(&str, &[&str]); 5] = [
	("size", &["bytes", "asserts", "decls", "defs", "checks", "nops", "bvlits", "other"]),
	("width", &["bits", "wmax", "wdist"]),
	("depth", &["depth"]),
	("product", &["mulrat", "bvmul", "bvurem", "bvudiv", "bvsdiv", "bvsrem", "bvsmod"]),
	("array", &["arrays", "select", "store"]),
];

/* the features of a group among those kept, by their names; "operators" takes the rest */
pub fn
group(d: &Data, kept: &[usize], name: &str) -> Vec<usize>
{
	let named = |names: &[&str]| kept.iter().copied().filter(|j| names.contains(&d.feats[*j].as_str())).collect();
	match(GROUPS.iter().find(|g| g.0 == name)){
		Some((_, names)) => named(names),
		None => kept.iter().copied().filter(|j| !GROUPS.iter().any(|g| g.1.contains(&d.feats[*j].as_str()))).collect(),
	}
}

/* a family at one setting over some of the features, taught by the training rows and
 * judged on the validation rows: the error of the time, or the mean f1 */
pub fn
worth(d: &Data, task: usize, train: &[usize], val: &[usize], f: usize, p: &[f64], keep: &[usize]) -> f64
{
	let t = &TASKS[task];
	let mut a = prep::mat(d, train, keep, t.classes);
	let mut b = prep::mat(d, val, keep, t.classes);
	prep::scale(&mut a, &mut b);
	let model = (families()[f].fit)(p, &a);
	judge(t.classes, &truth(d, t, val), &ask(&*model, &b))
}

/* what each feature is worth to a family at one setting, taught by the training rows:
 * the score lost on the validation rows when the feature is shuffled among them, a few times over.
 * worth reading where the family holds: a family that does not transfer loses as much
 * to any feature at all */
pub fn
importance(d: &Data, task: usize, train: &[usize], val: &[usize], f: usize, p: &[f64], repeats: usize) -> Vec<(String, f64)>
{
	let t = &TASKS[task];
	let fam = &families()[f];
	let kept = prep::keep(d, train);
	let mut a = prep::mat(d, train, &kept, t.classes);
	let mut b = prep::mat(d, val, &kept, t.classes);
	prep::scale(&mut a, &mut b);
	let model = (fam.fit)(p, &a);
	let want = truth(d, t, val);
	let plain = judge(t.classes, &want, &ask(&*model, &b));
	let mut rand = Rand::new(99);
	let mut worth = Vec::new();
	for j in 0..b.d{
		let mut lost = 0.0;
		for _ in 0..repeats{
			let mut col: Vec<f64> = (0..b.n).map(|i| b.x[i * b.d + j]).collect();
			rand.shuffle(&mut col);
			for i in 0..b.n{
				b.x[i * b.d + j] = col[i];
			}
			let s = judge(t.classes, &want, &ask(&*model, &b));
			lost += if(t.classes){ plain - s } else { s - plain };
		}
		worth.push((d.feats[kept[j]].to_string(), lost / repeats as f64));
	}
	worth.sort_by(|a, b| b.1.total_cmp(&a.1));
	worth
}

pub fn
num(x: f64) -> String
{
	format!("{:.4}", x)
}

fn
save(dir: &Path, name: &str, text: &str) -> Result<(), String>
{
	std::fs::write(dir.join(name), text).map_err(|e| format!("{}: {}", name, e))
}

/* every experiment to tables of text in dir: what the settings scored under cross
 * validation, how the chosen ones fared, what they predicted, and the parts */
pub fn
tables(d: &Data, exps: &[Exp], dir: &Path) -> Result<(), String>
{
	std::fs::create_dir_all(dir).map_err(|e| format!("{}: {}", dir.display(), e))?;
	let fams = families();
	let (mut cv, mut time, mut outcome, mut pred, mut conf, mut split) = (String::new(), String::new(), String::new(), String::new(), String::new(), String::new());
	cv.push_str("scheme\ttask\tmodel\tparams\tscore\n");
	time.push_str("scheme\ttask\tmodel\tparams\tval_mae\tval_rmse\tval_rho\tval_r2\ttest_mae\ttest_rmse\ttest_rho\ttest_r2\n");
	outcome.push_str("scheme\ttask\tmodel\tparams\tval_acc\tval_f1\tval_f1timeout\ttest_acc\ttest_f1\ttest_f1timeout\n");
	pred.push_str("scheme\ttask\tmodel\tset\tid\ttruth\tpred\n");
	conf.push_str("scheme\tmodel\ttruth\tpred\tn\n");
	split.push_str("scheme\ttask\tid\tpart\n");
	for e in exps{
		let t = &TASKS[e.task];
		for (rows, part) in [(&e.train, "train"), (&e.val, "val"), (&e.test, "test")]{
			for i in rows{
				split.push_str(&format!("{}\t{}\t{}\t{}\n", SCHEMES[e.scheme], t.name, d.id[*i], part));
			}
		}
		for (f, fam) in fams.iter().enumerate(){
			if(e.cv[f].is_empty()){
				continue;
			}
			let grid = if(t.classes){ &fam.gridc } else { &fam.gridr };
			for (cfg, s) in e.cv[f].iter().enumerate(){
				cv.push_str(&format!("{}\t{}\t{}\t{}\t{}\n", SCHEMES[e.scheme], t.name, fam.name, fam.show(&grid[cfg]), num(*s)));
			}
			let shown = e.shown(f, fam);
			let (a, b) = (e.valscore(d, f), e.testscore(d, f));
			if(t.classes){
				outcome.push_str(&format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n", SCHEMES[e.scheme], t.name, fam.name, shown,
					num(a.acc), num(a.f1), num(a.f1to), num(b.acc), num(b.f1), num(b.f1to)));
				for c in 0..CLASSES.len(){
					for g in 0..CLASSES.len(){
						conf.push_str(&format!("{}\t{}\t{}\t{}\t{}\n", SCHEMES[e.scheme], fam.name, CLASSES[c], CLASSES[g], b.conf.get(c, g)));
					}
				}
			}else{
				time.push_str(&format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n", SCHEMES[e.scheme], t.name, fam.name, shown,
					num(a.mae), num(a.rmse), num(a.rho), num(a.r2), num(b.mae), num(b.rmse), num(b.rho), num(b.r2)));
			}
			for (rows, set, p) in [(&e.val, "val", &e.valpred[f]), (&e.test, "test", &e.testpred[f])]{
				let truth = truth(d, t, rows);
				for i in 0..rows.len(){
					pred.push_str(&format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\n", SCHEMES[e.scheme], t.name, fam.name, set, d.id[rows[i]], truth[i], p[i]));
				}
			}
		}
	}
	save(dir, "cv.tsv", &cv)?;
	save(dir, "time.tsv", &time)?;
	save(dir, "outcome.tsv", &outcome)?;
	save(dir, "pred.tsv", &pred)?;
	save(dir, "conf.tsv", &conf)?;
	save(dir, "split.tsv", &split)?;
	Ok(())
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn scores()
	{
		let y = [1.0, 2.0, 3.0, 4.0];
		let s = score(false, &y, &[1.5, 2.0, 2.5, 4.5]);
		assert_eq!((s.mae, s.rho), (0.375, 1.0));
		assert_eq!((s.acc, s.f1), (0.0, 0.0));
		let s = score(true, &[0.0, 1.0, 2.0, 2.0], &[0.0, 1.0, 1.0, 2.0]);
		assert!((s.acc - 0.75).abs() < 1e-12);
		assert!((s.f1to - 2.0 / 3.0).abs() < 1e-12);
		assert_eq!(s.conf.get(2, 1), 1);
		assert_eq!(judge(true, &[0.0, 1.0, 2.0, 2.0], &[0.0, 1.0, 1.0, 2.0]), s.f1);
	}

	#[test]
	fn gate_holds_a_place()
	{
		let gate = Gate::new(2);
		let (mine, other) = (gate.enter(), gate.enter());
		assert!(gate.try_enter().is_none());
		drop(mine);
		assert!(gate.try_enter().is_some());
		/* a fit that comes apart gives its place back with its guard */
		let gone = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
			let _held = gate.enter();
			panic!("a fit gone wrong");
		}));
		assert!(gone.is_err());
		assert!(gate.try_enter().is_some());
		drop(other);
		assert_eq!(gate.0.load(Ordering::SeqCst), 2);
		/* and a thread after a place waits for one, then gets in */
		let gate = &gate;
		thread::scope(|scope| {
			let (done, seen) = mpsc::channel();
			let (both, mine) = (gate.enter(), gate.enter());
			scope.spawn(move || {
				let _held = gate.enter();
				done.send(()).ok();
			});
			assert!(seen.recv_timeout(Duration::from_millis(200)).is_err());
			drop(both);
			seen.recv_timeout(Duration::from_secs(5)).ok().unwrap();
			drop(mine);
		});
		assert_eq!(gate.0.load(Ordering::SeqCst), 2);
	}

	#[test]
	fn a_small_round_trip()
	{
		/* a toy dataset of a dozen families, where the time grows with the first feature and
		 * the class follows the second: under a split by family, the test part needs
		 * families of its own, so there are more of them than the parts could hold whole */
		let mut text = String::from("id\tlogic\tfamily\tstatus\tresult\twall\tcpu\tsize\tkind\tflat\n");
		let mut rand = Rand::new(5);
		for i in 0..120{
			let (size, kind) = (rand.next() % 9, rand.next() % 2);
			let (result, wall) = match((size, kind)){
				(8, _) => ("timeout", 1201.0),
				(s, 1) => ("sat", 10.0f64.powi(s as i32 - 3)),
				_ => ("unsat", 10.0f64.powi(size as i32 - 4)),
			};
			text.push_str(&format!("f{i}/L/f{}\tL\tf{}\tsat\t{}\t{}\t{}\t{}\t{}\t7\n", i % 12, i % 12, result, wall, wall, size, kind));
		}
		let d = Data::read(&crate::table::Table::read(text.as_bytes()).unwrap(), 1200.0).unwrap();
		let setup = Setup { seed: 1, threads: 2, places: 2, limit: 1200.0 };
		let gate = Gate::new(setup.places);
		let dir = std::env::temp_dir().join(format!("smtfit-test-{}", std::process::id()));
		let mut exps = Vec::new();
		for scheme in 0..SCHEMES.len(){
			for task in 0..TASKS.len(){
				exps.push(one(&d, &setup, scheme, task, &gate, &mut |_| {}));
			}
		}
		tables(&d, &exps, &dir).unwrap();
		for name in ["cv.tsv", "time.tsv", "outcome.tsv", "pred.tsv", "conf.tsv", "split.tsv"]{
			let text = std::fs::read_to_string(dir.join(name)).unwrap();
			assert!(text.lines().count() > 1, "{} is empty", name);
		}
		/* the parts cover every instance once, and no part is empty */
		let e = &exps[0];
		let mut all = [&e.train[..], &e.val[..], &e.test[..]].concat();
		all.sort_unstable();
		assert_eq!(all, (0..d.n).collect::<Vec<usize>>());
		for e in &exps{
			assert!(e.train.len() > 0 && e.val.len() > 0 && e.test.len() > 0, "{}/{}", SCHEMES[e.scheme], TASKS[e.task].name);
		}
		/* under a split by family, no family straddles two parts */
		let e = &exps[0];
		for i in &e.train{
			assert!(!e.val.contains(i) && !e.test.contains(i), "family {} straddles two parts", d.families[d.group[*i]]);
		}
		/* a tree settles tasks this clean under either split; the guesses do not */
		let named = |name: &str| families().iter().position(|f| f.name == name).unwrap();
		for e in &exps{
			let (tree, guess) = (named("tree"), named(if(TASKS[e.task].classes){ "majority" } else { "median" }));
			let (a, b) = (e.testscore(&d, tree), e.testscore(&d, guess));
			let better = if(TASKS[e.task].classes){ a.f1 > b.f1 } else { a.mae < b.mae };
			assert!(better, "{}/{}: the tree does not beat the guess", SCHEMES[e.scheme], TASKS[e.task].name);
		}
		/* to a tree under the split at random, the size of the formula is worth more than its
		 * kind, and the feature that never varies is no feature at all */
		let e = &exps[3];
		let worth = importance(&d, 0, &e.train, &e.val, named("tree"), &[10.0], 2);
		assert!(!worth.iter().any(|(f, _)| f == "flat"));
		let rank = |name: &str| worth.iter().position(|(f, _)| f == name).unwrap();
		assert!(rank("size") < rank("kind"), "{:?}", worth);
		std::fs::remove_dir_all(&dir).unwrap();
	}
}
