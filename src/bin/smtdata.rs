/* smtdata: the dataset, from competition results and the features of their formulas */
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufWriter, Write};

use smtdif::buf::Buf;
use smtdif::flags::Flags;
use smtdif::json::{Json, Tok};
use smtdif::stream::stream;
use smtdif::table::{col, Table};
use smtdif::{die, input, Res};

/* one line of the competition results */
#[derive(Default)]
struct Run {
	solver: String,
	logic: String,
	family: Vec<String>,
	name: String,
	result: String,
	wall: f64,
	cpu: f64,
}

impl Run {
	fn
	id(&self) -> String
	{
		format!("{}/{}/{}", self.logic, self.family.join("/"), self.name)
	}
}

/* the outcome of a run, as the class it falls in; none when the solver gave no answer */
fn
label(result: &str) -> Option<&'static str>
{
	match(result){
		"sat" => Some("sat"),
		"unsat" => Some("unsat"),
		/* out of memory is the time limit's twin:
		 * the resource ran out, not the solver's ideas */
		"Timeout" | "OutOfMemory" => Some("timeout"),
		_ => None,
	}
}

/* the place of a class among sat, unsat and timeout */
fn
class(lab: &str) -> usize
{
	["sat", "unsat", "timeout"].iter().position(|c| *c == lab).unwrap_or(0)
}

/* what became of the results read */
#[derive(Default)]
struct Tally {
	read: usize,
	solvers: usize,
	logics: usize,
	off: usize,
	features: usize,
	kept: [usize; 3],
}

fn
run<R: io::Read>(fl: &mut Flags, input: R, w: &mut impl Write) -> Res<()>
{
	if(fl.all('f').is_empty()){
		fl.usage("[-s solver] [-l logic]... [-f feats.tsv]... [results.json.gz]");
	}
	/* every formula seen, by benchmark */
	let mut index: HashMap<String, Vec<String>> = HashMap::new();
	let mut cols: Vec<String> = Vec::new();
	for path in fl.all('f'){
		let t = Table::read(File::open(path).map_err(|e| format!("can't open {}: {}", path, e))?)
			.map_err(|e| format!("{}: {}", path, e))?;
		if(cols.is_empty()){
			cols = t.cols.clone();
		}else if(t.cols != cols){
			return Err(format!("{}: its columns differ from those of the first table", path));
		}
		let ci = t.col("id").map_err(|e| format!("{}: {}", path, e))?;
		for r in t.rows{
			index.insert(r[ci].clone(), r);
		}
	}
	let ci = col(&cols, "id")?;
	let cl = col(&cols, "logic")?;
	let cs = col(&cols, "status")?;

	let mut j = Json::new(Buf::new(stream(input).map_err(|e| format!("results: {}", e))?));
	j.expect(Tok::Obj).map_err(|e| format!("results: {}", e))?;
	if(j.string()? != "results"){
		return Err("results: the key results wanted".to_string());
	}
	j.expect(Tok::Arr)?;

	let solver = fl.get('s');
	let logics = fl.all('l');
	let mut tally = Tally::default();
	let mut seen: HashSet<String> = HashSet::new();
	let mut rows: Vec<(String, Vec<String>)> = Vec::new();
	loop {
		match(j.tok()?){
			Tok::Endarr => break,
			Tok::Obj => {
				tally.read += 1;
				let r = one(&mut j)?;
				if(solver.map(|s| s != r.solver).unwrap_or(false)){
					tally.solvers += 1;
				}else if(!logics.is_empty() && !logics.iter().any(|l| *l == r.logic)){
					tally.logics += 1;
				}else if(label(&r.result).is_none()){
					tally.off += 1;
				}else{
					let id = r.id();
					match(index.get(&id)){
						None => tally.features += 1,
						Some(f) => {
							if(seen.insert(id.clone())){
								let lab = label(&r.result).unwrap_or("");
								tally.kept[class(lab)] += 1;
								let mut row = Vec::with_capacity(7 + cols.len() - 3);
								row.push(id.clone());
								row.push(r.logic.clone());
								row.push(r.family.first().cloned().unwrap_or_default());
								row.push(f[cs].clone());
								row.push(lab.to_string());
								row.push(r.wall.to_string());
								row.push(r.cpu.to_string());
								for (k, v) in f.iter().enumerate(){
									if(k != ci && k != cl && k != cs){
										row.push(v.clone());
									}
								}
								rows.push((id, row));
							}
						}
					}
				}
			}
			_ => return Err("results: an object wanted".to_string()),
		}
	}
	j.expect(Tok::Endobj)?;
	j.end()?;

	rows.sort_by(|a, b| a.0.cmp(&b.0));
	let mut out = String::from("id\tlogic\tfamily\tstatus\tresult\twall\tcpu");
	for (k, c) in cols.iter().enumerate(){
		if(k != ci && k != cl && k != cs){
			out.push('\t');
			out.push_str(c);
		}
	}
	out.push('\n');
	for (_, r) in &rows{
		out.push_str(&r.join("\t"));
		out.push('\n');
	}
	w.write_all(out.as_bytes()).map_err(|e| format!("write: {}", e))?;
	eprintln!(
		"{}: {} results; {} other solvers; {} other logics; {} without an outcome; {} without features; {} kept: sat {}, unsat {}, timeout {}",
		fl.name(), tally.read, tally.solvers, tally.logics, tally.off, tally.features,
		rows.len(), tally.kept[0], tally.kept[1], tally.kept[2]);
	Ok(())
}

/* one object of the results array */
fn
one<R: io::Read>(j: &mut Json<R>) -> Res<Run>
{
	let mut r = Run::default();
	loop {
		match(j.tok()?){
			Tok::Endobj => return Ok(r),
			Tok::Str(k) => match(k.as_str()){
				"solver" => r.solver = j.string()?,
				"result" => r.result = j.string()?,
				"wallclock_time" => r.wall = j.num()?,
				"cpu_time" => r.cpu = j.num()?,
				"file" => described(j, &mut r)?,
				_ => j.skip()?,
			},
			_ => return Err("results: a key wanted".to_string()),
		}
	}
}

/* the file object of a result: the logic, the family folders, and the name */
fn
described<R: io::Read>(j: &mut Json<R>, r: &mut Run) -> Res<()>
{
	j.expect(Tok::Obj)?;
	loop {
		match(j.tok()?){
			Tok::Endobj => return Ok(()),
			Tok::Str(k) => match(k.as_str()){
				"logic" => r.logic = j.string()?,
				"name" => r.name = j.string()?,
				"family" => {
					j.expect(Tok::Arr)?;
					loop {
						match(j.tok()?){
							Tok::Endarr => break,
							Tok::Str(s) => r.family.push(s),
							_ => return Err("results: a folder name wanted".to_string()),
						}
					}
				}
				_ => j.skip()?,
			},
			_ => return Err("results: a key wanted".to_string()),
		}
	}
}

fn
main()
{
	let mut fl = Flags::new();
	fl.parse("");
	let name = fl.name().to_string();
	let mut w = BufWriter::new(io::stdout().lock());
	match input(&fl.argv).and_then(|input| run(&mut fl, input, &mut w)).and_then(|_| w.flush().map_err(|e| e.to_string())){
		Ok(()) => {}
		Err(e) => die(&name, &e),
	}
}
