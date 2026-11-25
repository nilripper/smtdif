/* tree: a cart tree grown on weighted rows, to fit a target or to sort classes */
use crate::model::{Mat, Model};
use crate::rng::Rand;
use crate::stat::argmax;

/* the rows of a matrix in the order of each feature: the same orders serve every tree */
pub struct Orders {
	pub n: usize,
	pub d: usize,
	/* d runs of n; ord[f * n + p] is the row pth lowest on feature f */
	ord: Vec<usize>,
}

impl Orders {
	pub fn
	new(x: &[f64], n: usize, d: usize) -> Orders
	{
		let mut ord = vec![0usize; d * n];
		for f in 0..d{
			let o = &mut ord[f * n..f * n + n];
			for p in 0..n{
				o[p] = p;
			}
			o.sort_by(|a, b| x[a * d + f].total_cmp(&x[b * d + f]));
		}
		Orders { n, d, ord }
	}
}

/* what a tree is grown from */
pub struct Cut<'a> {
	pub x: &'a [f64],
	pub d: usize,
	/* the target, in a regression; left alone in a classification */
	pub y: &'a [f64],
	pub c: &'a [usize],
	/* the classes; 0 in a regression */
	pub k: usize,
	/* a weight for each row: a bootstrap count, or one */
	pub w: &'a [f64],
	pub ord: &'a Orders,
	pub minleaf: usize,
	pub maxdepth: usize,
	/* the features drawn for each split; all of them when 0 */
	pub mtry: usize,
	pub rand: &'a mut Rand,
}

pub struct Tree {
	/* a node sends x[feat[i]] at or under thr[i] to kid[i][0], and the rest to kid[i][1].
	 * a leaf holds usize::MAX for its feature, and in val[i] the fit, or the class */
	feat: Vec<usize>,
	thr: Vec<f64>,
	kid: Vec<[usize; 2]>,
	val: Vec<f64>,
}

impl Model for Tree {
	fn
	predict(&self, x: &[f64]) -> f64
	{
		let mut i = 0;
		while(self.feat[i] != usize::MAX){
			i = self.kid[i][usize::from(x[self.feat[i]] > self.thr[i])];
		}
		self.val[i]
	}
}

/* a tree at work: the orders of its open node, parted as it splits, and the tree so far */
struct Grow<'a> {
	cut: Cut<'a>,
	ord: Vec<usize>,
	/* which rows of the open node go left */
	left: Vec<bool>,
	/* room to part a run without disturbing the order on either side */
	buf: Vec<usize>,
	tree: Tree,
}

/* the best place met to divide a run: its score, and where */
type Split = (f64, usize);

/* the tree over the rows of the cut */
pub fn
cart(cut: Cut) -> Tree
{
	let n = cut.ord.n;
	let sorted = cut.ord.ord.clone();
	let mut g = Grow {
		cut,
		ord: sorted,
		left: vec![false; n],
		buf: vec![0; n],
		tree: Tree { feat: Vec::new(), thr: Vec::new(), kid: Vec::new(), val: Vec::new() },
	};
	g.node(0, n, 0);
	g.tree
}

impl<'a> Grow<'a> {
	fn
	x(&self, row: usize, f: usize) -> f64
	{
		self.cut.x[row * self.cut.d + f]
	}

	/* the fit, or the class, of a node: over the rows its run holds, whatever their order */
	fn
	leaf(&self, a: usize, b: usize) -> f64
	{
		let w = &self.cut.w;
		if(self.cut.k == 0){
			let (mut sw, mut swy) = (0.0, 0.0);
			for p in a..b{
				let r = self.ord[p];
				sw += w[r];
				swy += w[r] * self.cut.y[r];
			}
			return swy / sw;
		}
		let mut share = vec![0.0; self.cut.k];
		for p in a..b{
			share[self.cut.c[self.ord[p]]] += w[self.ord[p]];
		}
		argmax(&share) as f64
	}

	fn
	best(&self, f: usize, a: usize, b: usize) -> Option<Split>
	{
		let n = self.cut.ord.n;
		let row = |p: usize| self.ord[f * n + p];
		let val = |p: usize| self.x(row(p), f);
		let least = self.cut.minleaf as f64;
		let mut best: Option<Split> = None;
		let better = |s: f64, b: &Option<Split>| b.as_ref().map_or(true, |b| s < b.0);
		if(self.cut.k == 0){
			/* the errors left on each side, from running sums of w, wy and wy squared */
			let (mut tw, mut twy, mut twyy) = (0.0, 0.0, 0.0);
			for p in a..b{
				let (wp, yp) = (self.cut.w[row(p)], self.cut.y[row(p)]);
				tw += wp;
				twy += wp * yp;
				twyy += wp * yp * yp;
			}
			let (mut sw, mut swy, mut swyy) = (0.0, 0.0, 0.0);
			for mid in a + 1..b{
				let (wp, yp) = (self.cut.w[row(mid - 1)], self.cut.y[row(mid - 1)]);
				sw += wp;
				swy += wp * yp;
				swyy += wp * yp * yp;
				if(val(mid - 1) == val(mid) || sw < least || tw - sw < least){
					continue;
				}
				let s = swyy - swy * swy / sw + (twyy - swyy) - (twy - swy) * (twy - swy) / (tw - sw);
				if(better(s, &best)){
					best = Some((s, mid));
				}
			}
		}else{
			/* the gini impurity of both sides falls as their squared class shares rise */
			let k = self.cut.k;
			let mut tot = vec![0.0; k];
			for p in a..b{
				tot[self.cut.c[row(p)]] += self.cut.w[row(p)];
			}
			let all: f64 = tot.iter().sum();
			let mut left = vec![0.0; k];
			for mid in a + 1..b{
				let p = row(mid - 1);
				left[self.cut.c[p]] += self.cut.w[p];
				let sw = left.iter().sum::<f64>();
				if(val(mid - 1) == val(mid) || sw < least || all - sw < least){
					continue;
				}
				let gain = left.iter().map(|v| v * v).sum::<f64>() / sw
					+ tot.iter().zip(&left).map(|(t, l)| (t - l) * (t - l)).sum::<f64>() / (all - sw);
				if(better(-gain, &best)){
					best = Some((-gain, mid));
				}
			}
		}
		best
	}

	/* part the run a..b of feature g's orders at mid: the rows marked left to the front,
	 * each side keeping its order, for the run is sorted and must stay so */
	fn
	part(&mut self, g: usize, a: usize, b: usize, mid: usize)
	{
		let Grow { ord, left, buf, cut, .. } = self;
		let n = ord.len() / cut.d;
		let run = &mut ord[g * n..(g + 1) * n];
		let (mut l, mut r) = (a, mid);
		for p in a..b{
			if(left[run[p]]){
				buf[l] = run[p];
				l += 1;
			}else{
				buf[r] = run[p];
				r += 1;
			}
		}
		run[a..b].copy_from_slice(&buf[a..b]);
	}

	/* the node over the orders a..b, at its depth, and its index */
	fn
	node(&mut self, a: usize, b: usize, depth: usize) -> usize
	{
		let me = self.tree.feat.len();
		let leaf = self.leaf(a, b);
		self.tree.feat.push(usize::MAX);
		self.tree.thr.push(0.0);
		self.tree.kid.push([me; 2]);
		self.tree.val.push(leaf);
		let weight: f64 = (a..b).map(|p| self.cut.w[self.ord[p]]).sum();
		if(depth >= self.cut.maxdepth || weight < 2.0 * self.cut.minleaf as f64){
			return me;
		}
		/* the features this node may split on */
		let every = self.cut.mtry == 0 || self.cut.mtry >= self.cut.d;
		let mut cands: Vec<usize> = (0..self.cut.d).collect();
		if !every{
			self.cut.rand.shuffle(&mut cands);
			cands.truncate(self.cut.mtry);
			cands.sort_unstable();
		}
		let mut best: Option<(f64, usize, usize)> = None;
		for f in cands{
			if let Some((s, mid)) = self.best(f, a, b){
				if(best.as_ref().map_or(true, |b| s < b.0)){
					best = Some((s, f, mid));
				}
			}
		}
		let Some((_, f, mid)) = best else {
			return me;
		};
		let n = self.cut.ord.n;
		let last = self.x(self.ord[f * n + mid - 1], f);
		let next = self.x(self.ord[f * n + mid], f);
		/* the rows under the threshold: every feature's run parts the same way */
		for p in a..mid{
			self.left[self.ord[f * n + p]] = true;
		}
		for g in 0..self.cut.d{
			self.part(g, a, b, mid);
		}
		/* the same rows, now at the front of every run */
		for p in a..mid{
			self.left[self.ord[f * n + p]] = false;
		}
		let l = self.node(a, mid, depth + 1);
		let r = self.node(mid, b, depth + 1);
		self.tree.feat[me] = f;
		self.tree.thr[me] = (last + next) / 2.0;
		self.tree.kid[me] = [l, r];
		me
	}
}

/* a lone tree over all the rows at their full weight, as deep as minleaf lets */
pub fn
treefit(p: &[f64], m: &Mat) -> Box<dyn Model>
{
	let orders = Orders::new(&m.x, m.n, m.d);
	let ones = vec![1.0; m.n];
	let mut rand = Rand::new(0);
	Box::new(cart(Cut {
		x: &m.x,
		d: m.d,
		y: &m.y,
		c: &m.c,
		k: m.k,
		w: &ones,
		ord: &orders,
		minleaf: p[0] as usize,
		maxdepth: 64,
		mtry: 0,
		rand: &mut rand,
	}))
}

#[cfg(test)]
mod test {
	use super::*;

	fn
	close(a: f64, b: f64)
	{
		assert!((a - b).abs() < 1e-9, "{} is not {}", a, b);
	}

	/* n rows along a line, d features each at x = i + j, with the target f(i) */
	fn
	line(n: usize, d: usize, f: impl Fn(usize) -> f64) -> (Vec<f64>, Vec<f64>)
	{
		let (mut x, mut y) = (Vec::new(), Vec::new());
		for i in 0..n{
			x.extend((0..d).map(|j| (i + j) as f64));
			y.push(f(i));
		}
		(x, y)
	}

	fn
	grow(x: &[f64], y: &[f64], c: &[usize], k: usize, w: &[f64], minleaf: usize, maxdepth: usize) -> Tree
	{
		let n = y.len();
		let d = x.len() / n;
		let orders = Orders::new(x, n, d);
		let mut rand = Rand::new(0);
		cart(Cut { x, d, y, c, k, w, ord: &orders, minleaf, maxdepth, mtry: 0, rand: &mut rand })
	}

	#[test]
	fn orders()
	{
		/* the rows are [3, 9], [1, 7] and [2, 8]: by the first feature they fall 1, 2, 0; by the second, 1, 2, 0 too */
		let o = Orders::new(&[3.0, 9.0, 1.0, 7.0, 2.0, 8.0], 3, 2);
		assert_eq!(o.ord, [1, 2, 0, 1, 2, 0]);
	}

	#[test]
	fn a_curved_target()
	{
		/* two features that rise and fall along the rows, so that no run keeps the row order:
		 * a tree with single row leaves still fits the training rows one by one */
		let n = 400;
		let mut m = Mat { n, d: 2, k: 0, x: Vec::new(), y: Vec::new(), c: Vec::new() };
		for i in 0..n{
			let t = i as f64 / n as f64 * 6.0 * std::f64::consts::PI;
			let (a, b) = (t.cos() * 3.0, (t / 2.0).sin() * 3.0);
			m.x.extend_from_slice(&[a, b]);
			m.y.push(a * b / 10.0);
			m.c.push(0);
		}
		let t = treefit(&[1.0], &m);
		for i in 0..n{
			close(t.predict(m.row(i)), m.y[i]);
		}
		let t = treefit(&[20.0], &m);
		let err = (0..n).map(|i| (t.predict(m.row(i)) - m.y[i]).abs()).sum::<f64>() / n as f64;
		assert!(err < 0.15, "mean error {}", err);
	}

	#[test]
	fn a_step_function()
	{
		/* the target steps with x: three steps over 300 rows */
		let (x, y) = line(300, 2, |i| match(i){
			0..=99 => 1.0,
			100..=199 => 5.0,
			_ => -2.0,
		});
		for minleaf in [1, 5, 20]{
			let t = grow(&x, &y, &[0; 300], 0, &vec![1.0; 300], minleaf, 64);
			for i in 0..300{
				close(t.predict(&x[i * 2..i * 2 + 2]), y[i]);
			}
			assert_eq!(t.predict(&[150.0, 151.0]), 5.0);
		}
	}

	#[test]
	fn least_leaf_binds()
	{
		/* a lone row far off would have a leaf of its own, were one row enough */
		let (x, y) = line(20, 1, |i| if(i == 0){ 100.0 } else { (i % 2) as f64 });
		let t = grow(&x, &y, &[0; 20], 0, &vec![1.0; 20], 5, 64);
		assert!(t.predict(&[0.0]) < 50.0);
		let t = grow(&x, &y, &[0; 20], 0, &vec![1.0; 20], 1, 64);
		close(t.predict(&[0.0]), 100.0);
	}

	#[test]
	fn classes()
	{
		let (x, _) = line(60, 2, |_| 0.0);
		let c: Vec<usize> = (0..60).map(|i| match(i % 10){
			0..=3 => 0,
			4..=6 => 1,
			_ => 2,
		}).collect();
		let t = grow(&x, &[0.0; 60], &c, 3, &vec![1.0; 60], 1, 64);
		for i in 0..60{
			assert_eq!(t.predict(&x[i * 2..i * 2 + 2]) as usize, c[i]);
		}
	}

	#[test]
	fn weights_and_depth()
	{
		/* with the odd rows weighed nothing, the tree sees only the even ones; an odd
		 * row falls in with an even neighbor, and takes its value */
		let (x, y) = line(10, 1, |i| if(i % 2 == 0){ -(i as f64) } else { 1000.0 });
		let w: Vec<f64> = (0..10).map(|i| f64::from(i % 2 == 0)).collect();
		let t = grow(&x, &y, &[0; 10], 0, &w, 1, 64);
		for i in 0..10{
			let p = t.predict(&[i as f64]);
			if(i % 2 == 0){
				close(p, -(i as f64));
			}else{
				assert!(p == -(i as f64 - 1.0) || p == -(i as f64 + 1.0), "row {} gave {}", i, p);
			}
		}
		/* one level deep, a rising line parts at its middle: the means of 0..=4 and of 5..=9 */
		let (_, y) = line(10, 1, |i| i as f64);
		let t = grow(&x, &y, &[0; 10], 0, &vec![1.0; 10], 1, 1);
		close(t.predict(&[0.0]), 2.0);
		close(t.predict(&[9.0]), 7.0);
	}
}
