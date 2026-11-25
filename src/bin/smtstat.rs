/* smtstat: what the dataset holds, in tables of text */
use std::collections::HashSet;
use std::fmt::Write as _;
use std::io::{self, Read, Write};

use smtdif::data::{Data, CLASSES};
use smtdif::flags::Flags;
use smtdif::stat::{mean, median, quantile, sd, spearman};
use smtdif::table::Table;
use smtdif::{die, input, Res};

/* a line onto out; a string never refuses one */
macro_rules! say {
	($out:expr, $($arg:tt)*) => {
		let _ = writeln!($out, $($arg)*);
	};
}

fn
num(x: f64) -> String
{
	format!("{:.4}", x)
}

/* min, first quartile, median, third quartile and max of a sample */
fn
spread(v: &[f64]) -> [f64; 5]
{
	let mut s = v.to_vec();
	s.sort_by(|a, b| a.total_cmp(b));
	[quantile(&s, 0.0), quantile(&s, 0.25), quantile(&s, 0.5), quantile(&s, 0.75), quantile(&s, 1.0)]
}

fn
run<R: Read>(fl: &mut Flags, input: R, out: &mut String) -> Res<()>
{
	let limit = fl.num('t', 1200.0)?;
	let t = Table::read(input).map_err(|e| format!("dataset: {}", e))?;
	let d = Data::read(&t, limit)?;
	let cw = t.col("wall")?;
	let walls: Vec<f64> = t.rows.iter().map(|r| r[cw].parse().unwrap_or(0.0)).collect();
	let solved: Vec<usize> = (0..d.n).filter(|i| d.solved[*i]).collect();
	let timeouts: Vec<f64> = d.solved.iter().map(|s| u8::from(!*s) as f64).collect();

	/* every column sorted into text or number, with its sample and its empty cells */
	let mut text: Vec<(String, usize, usize)> = Vec::new();
	let mut nums: Vec<(String, Vec<f64>, usize)> = Vec::new();
	for (k, c) in t.cols.iter().enumerate(){
		let (mut v, mut miss, mut bad) = (Vec::new(), 0, 0);
		for r in &t.rows{
			match(r[k].parse::<f64>()){
				Ok(x) => v.push(x),
				Err(_) if r[k].is_empty() => miss += 1,
				Err(_) => bad += 1,
			}
		}
		if(bad > 0){
			let values: HashSet<&str> = t.rows.iter().map(|r| r[k].as_str()).collect();
			text.push((c.clone(), values.len(), miss));
		}else if(c != "id"){
			nums.push((c.clone(), v, miss));
		}
	}
	let empty: usize = text.iter().map(|c| c.2).chain(nums.iter().map(|c| c.2)).sum();

	say!(out, "# the dataset, by count");
	say!(out, "what\tcount");
	say!(out, "instances\t{}", d.n);
	say!(out, "features\t{}", d.d);
	say!(out, "families\t{}", d.families.len());
	say!(out, "empty cells\t{}", empty);

	say!(out, "\n# text columns: the values each takes");
	say!(out, "column\tvalues\tempty");
	for (name, values, miss) in &text{
		say!(out, "{}\t{}\t{}", name, values, miss);
	}

	say!(out, "\n# numeric columns, the time in seconds and in its base 10 logarithm; a timeout counts twice the limit");
	say!(out, "column\tn\tempty\tmin\tq25\tmedian\tq75\tmax\tmean\tsd");
	nums.push(("log10 wall".to_string(), d.y.clone(), 0));
	nums.push(("log10 wall of the solved".to_string(), solved.iter().map(|i| d.y[*i]).collect(), 0));
	nums.push(("wall of the solved".to_string(), solved.iter().map(|i| walls[*i]).collect(), 0));
	nums.push(("timeout".to_string(), timeouts.clone(), 0));
	for (name, v, miss) in &nums{
		let s = spread(v);
		say!(out, "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", name, v.len(), miss,
			num(s[0]), num(s[1]), num(s[2]), num(s[3]), num(s[4]), num(mean(v)), num(sd(v)));
	}

	say!(out, "\n# log10 wall by bins a tenth wide: where each bin starts, the instances in it, the timeouts among them");
	say!(out, "from\tinstances\ttimeouts");
	let lo = d.y.iter().cloned().fold(f64::MAX, f64::min);
	let hi = d.y.iter().cloned().fold(f64::MIN, f64::max);
	let bins = ((hi - lo) * 10.0).ceil() as usize + 1;
	let (mut count, mut to) = (vec![0usize; bins], vec![0usize; bins]);
	for i in 0..d.n{
		let b = ((d.y[i] - lo) * 10.0).floor() as usize;
		count[b] += 1;
		to[b] += usize::from(!d.solved[i]);
	}
	for b in 0..bins{
		if(count[b] > 0){
			say!(out, "{}\t{}\t{}", num(lo + b as f64 / 10.0), count[b], to[b]);
		}
	}

	say!(out, "\n# outcomes by logic");
	say!(out, "logic\tsat\tunsat\ttimeout\ttotal\ttimeout share");
	let mut logics: Vec<String> = d.logic.clone();
	logics.sort();
	logics.dedup();
	logics.push("all".to_string());
	for logic in &logics{
		let mut n = [0usize; 3];
		for i in 0..d.n{
			if(logic == "all" || &d.logic[i] == logic){
				n[d.c[i]] += 1;
			}
		}
		let total: usize = n.iter().sum();
		say!(out, "{}\t{}\t{}\t{}\t{}\t{}", logic, n[0], n[1], n[2], total, num(n[2] as f64 / total as f64));
	}
	say!(out, "\n# the classes, by the number the tools give them");
	say!(out, "number\tclass\tinstances");
	for c in 0..CLASSES.len(){
		say!(out, "{}\t{}\t{}", c, CLASSES[c], d.c.iter().filter(|x| **x == c).count());
	}

	say!(out, "\n# families, largest first; the time is the median wall clock of the solved instances");
	say!(out, "family\tlogics\tinstances\tsat\tunsat\ttimeout\tmedian wall");
	let mut order: Vec<usize> = (0..d.families.len()).collect();
	order.sort_by_key(|g| std::cmp::Reverse(d.group.iter().filter(|x| **x == *g).count()));
	for g in order{
		let members: Vec<usize> = (0..d.n).filter(|i| d.group[*i] == g).collect();
		let mut kinds: Vec<String> = members.iter().map(|i| d.logic[*i].clone()).collect();
		kinds.sort();
		kinds.dedup();
		let mut n = [0usize; 3];
		for i in &members{
			n[d.c[*i]] += 1;
		}
		let times: Vec<f64> = members.iter().filter(|i| d.solved[**i]).map(|i| walls[*i]).collect();
		say!(out, "{}\t{}\t{}\t{}\t{}\t{}\t{}", d.families[g], kinds.join(" "), members.len(), n[0], n[1], n[2], num(median(&times)));
	}

	say!(out, "\n# spearman correlation of each feature with the targets");
	say!(out, "feature\tlog10 wall\tlog10 wall of the solved\ttimeout");
	let ys: Vec<f64> = solved.iter().map(|i| d.y[*i]).collect();
	for k in 0..d.d{
		let all: Vec<f64> = (0..d.n).map(|i| d.x[i * d.d + k]).collect();
		let part: Vec<f64> = solved.iter().map(|i| d.x[i * d.d + k]).collect();
		say!(out, "{}\t{}\t{}\t{}", d.feats[k], num(spearman(&all, &d.y)), num(spearman(&part, &ys)), num(spearman(&all, &timeouts)));
	}
	Ok(())
}

fn
main()
{
	let mut fl = Flags::new();
	fl.parse("");
	let name = fl.name().to_string();
	let mut out = String::new();
	let res = input(&fl.argv).and_then(|input| run(&mut fl, input, &mut out))
		.and_then(|_| io::stdout().lock().write_all(out.as_bytes()).map_err(|e| format!("write: {}", e)));
	match res{
		Ok(()) => {}
		Err(e) => die(&name, &e),
	}
}
