/* smtcal: what a few instances of a family new to the model are worth to its predictions.
 * the first k instances of each family, in the order of their names, set the family's
 * offset: the median of what the model missed on them. the rest are judged with the
 * offset added, so that k = 0 judges the predictions as they came */
use std::collections::{HashMap, HashSet};
use std::io::{self, BufWriter, Read, Write};

use smtdif::flags::Flags;
use smtdif::stat::{mae, median, rmse, spearman};
use smtdif::table::Table;
use smtdif::{die, input, Res};

/* one prediction of one row */
struct Guess {
	id: String,
	truth: f64,
	pred: f64,
}

fn
run<R: Read>(fl: &mut Flags, input: R, w: &mut impl Write) -> Res<()>
{
	let mut ks: Vec<usize> = Vec::new();
	for k in fl.get('k').unwrap_or("0,1,2,5,10,20,50").split(','){
		ks.push(k.trim().parse::<usize>().map_err(|_| format!("-k: {:?} is not a whole number", k))?);
	}
	let t = Table::read(input).map_err(|e| format!("predictions: {}", e))?;
	let (cs, ct, cm, cset, cid, ctr, cp) = (t.col("scheme")?, t.col("task")?, t.col("model")?,
		t.col("set")?, t.col("id")?, t.col("truth")?, t.col("pred")?);
	/* the predictions on test, by split, task and model, and there by family */
	let mut groups: Vec<(String, String, String)> = Vec::new();
	let mut seen: HashSet<(String, String, String)> = HashSet::new();
	let mut timed: HashMap<(String, String, String), bool> = HashMap::new();
	let mut ofs: HashMap<(String, String, String, String), Vec<Guess>> = HashMap::new();
	for r in &t.rows{
		if(r[cset] != "test"){
			continue;
		}
		let key = (r[cs].to_string(), r[ct].to_string(), r[cm].to_string());
		if(seen.insert(key.clone())){
			groups.push(key.clone());
		}
		let truth = r[ctr].parse::<f64>().map_err(|_| format!("{}: truth is not a number", r[cid]))?;
		let pred = r[cp].parse::<f64>().map_err(|_| format!("{}: pred is not a number", r[cid]))?;
		/* a class has no offset: a task is one of the times once a truth of it is not a class */
		let entry = timed.entry(key.clone()).or_insert(false);
		*entry |= truth.fract() != 0.0 || !(0.0..8.0).contains(&truth);
		/* the family is the first folder under the logic */
		let family = r[cid].split('/').nth(1).unwrap_or("").to_string();
		ofs.entry((key.0, key.1, key.2, family)).or_default()
			.push(Guess { id: r[cid].clone(), truth, pred });
	}
	let mut out = String::from("scheme\ttask\tmodel\tk\trows\tmae\trmse\trho\trho_within\n");
	for key in &groups{
		if !timed[key]{
			continue;
		}
		let mut families: Vec<&mut Vec<Guess>> = Vec::new();
		for ((s, t, m, _), rows) in ofs.iter_mut(){
			if(s == &key.0 && t == &key.1 && m == &key.2){
				rows.sort_by(|a, b| a.id.cmp(&b.id));
				families.push(rows);
			}
		}
		families.sort_by(|a, b| a[0].id.cmp(&b[0].id));
		for k in &ks{
			/* what the instances after the first k missed, and by how much; and how well
			 * the model orders the instances of one family, which no offset changes */
			let (mut truth, mut pred) = (Vec::new(), Vec::new());
			let (mut within, mut ordered) = (0.0, 0.0);
			for rows in &families{
				if(rows.len() <= *k){
					continue;
				}
				let missed: Vec<f64> = rows[..*k].iter().map(|g| g.truth - g.pred).collect();
				let offset = median(&missed);
				let (own, got): (Vec<f64>, Vec<f64>) = rows[*k..].iter().map(|g| (g.truth, g.pred)).unzip();
				truth.extend_from_slice(&own);
				pred.extend(got.iter().map(|p| p + offset));
				if(own.len() >= 3){
					within += spearman(&own, &got) * own.len() as f64;
					ordered += own.len() as f64;
				}
			}
			let within = if(ordered > 0.0){ within / ordered } else { 0.0 };
			out.push_str(&format!("{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\n", key.0, key.1, key.2,
				k, truth.len(), mae(&truth, &pred), rmse(&truth, &pred), spearman(&truth, &pred), within));
		}
	}
	w.write_all(out.as_bytes()).map_err(|e| format!("write: {}", e))?;
	Ok(())
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
