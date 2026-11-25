/* forest: random forests, and gradient boosting over regression trees */
use crate::model::{Mat, Model};
use crate::rng::Rand;
use crate::stat::{argmax, mean};
use crate::tree::{cart, Cut, Orders, Tree};

/* fixed for every setting the cross validation tries */
static NTREES: usize = 300;
static ROUNDS: usize = 200;
static RATE: f64 = 0.1;
static BOOSTLEAF: usize = 20;

struct Forest {
	trees: Vec<Tree>,
	/* the classes; 0 when the forest regresses */
	k: usize,
}

impl Model for Forest {
	fn
	predict(&self, x: &[f64]) -> f64
	{
		if(self.k == 0){
			return self.trees.iter().map(|t| t.predict(x)).sum::<f64>() / self.trees.len() as f64;
		}
		/* every tree has a vote */
		let mut votes = vec![0.0; self.k];
		for t in &self.trees{
			votes[t.predict(x) as usize] += 1.0;
		}
		argmax(&votes) as f64
	}
}

/* ntrees trees, each on a bootstrap sample of the rows, and on a draw of the features at every split */
pub fn
forest(p: &[f64], m: &Mat) -> Box<dyn Model>
{
	let orders = Orders::new(&m.x, m.n, m.d);
	let mtry = (m.d as f64).sqrt().floor().max(1.0) as usize;
	let mut trees = Vec::with_capacity(NTREES);
	for t in 0..NTREES{
		/* the bootstrap: as many draws from the rows as there are rows, counted in the weights */
		let mut rand = Rand::new(t as u64);
		let mut w = vec![0.0; m.n];
		for _ in 0..m.n{
			w[rand.below(m.n)] += 1.0;
		}
		trees.push(cart(Cut {
			x: &m.x,
			d: m.d,
			y: &m.y,
			c: &m.c,
			k: m.k,
			w: &w,
			ord: &orders,
			minleaf: p[0] as usize,
			maxdepth: 64,
			mtry,
			rand: &mut rand,
		}));
	}
	Box::new(Forest { trees, k: m.k })
}

struct Boost {
	/* a round of k trees for a classification, of one tree for a regression */
	trees: Vec<Tree>,
	k: usize,
	/* where a regression starts: the mean of the training targets */
	start: f64,
}

impl Model for Boost {
	fn
	predict(&self, x: &[f64]) -> f64
	{
		if(self.k == 0){
			return self.start + RATE * self.trees.iter().map(|t| t.predict(x)).sum::<f64>();
		}
		let mut score = vec![0.0; self.k];
		for (i, t) in self.trees.iter().enumerate(){
			score[i % self.k] += t.predict(x);
		}
		argmax(&score) as f64
	}
}

/* rounds of shallow trees, each fitted to what the ones before left out: the residual
 * in a regression; the gap between the class and its softmax share, one tree a class, otherwise */
pub fn
boost(p: &[f64], m: &Mat) -> Box<dyn Model>
{
	let orders = Orders::new(&m.x, m.n, m.d);
	let ones = vec![1.0; m.n];
	let mut rand = Rand::new(1);
	let depth = p[0] as usize;
	let cols = m.k.max(1);
	/* what the trees add to each row, target by target; and what one tree is left to fit */
	/* what the trees add to each row, target by target; and what one tree is left to fit.
	 * a regression starts at the mean of its targets, and the trees fit what is beyond it */
	let start = if(m.k == 0){ mean(&m.y) } else { 0.0 };
	let (mut fit, mut miss) = (vec![0.0; m.n * cols], vec![0.0; m.n * cols]);
	let mut trees: Vec<Tree> = Vec::with_capacity(ROUNDS * cols);
	for _ in 0..ROUNDS{
		for i in 0..m.n{
			if(m.k == 0){
				miss[i] = m.y[i] - start - fit[i];
			}else{
				let s = &fit[i * cols..i * cols + cols];
				let top = s.iter().cloned().fold(f64::MIN, f64::max);
				let sum = s.iter().map(|v| (v - top).exp()).sum::<f64>();
				for c in 0..cols{
					miss[i * cols + c] = f64::from(m.c[i] == c) - (s[c] - top).exp() / sum;
				}
			}
		}
		for c in 0..cols{
			let target: Vec<f64> = (0..m.n).map(|i| miss[i * cols + c]).collect();
			let tree = cart(Cut {
				x: &m.x,
				d: m.d,
				y: &target,
				c: &m.c,
				k: 0,
				w: &ones,
				ord: &orders,
				minleaf: BOOSTLEAF,
				maxdepth: depth,
				mtry: 0,
				rand: &mut rand,
			});
			for i in 0..m.n{
				fit[i * cols + c] += RATE * tree.predict(m.row(i));
			}
			trees.push(tree);
		}
	}
	Box::new(Boost { trees, k: m.k, start })
}

#[cfg(test)]
mod test {
	use super::*;

	fn
	close(a: f64, b: f64)
	{
		assert!((a - b).abs() < 1e-9, "{} is not {}", a, b);
	}

	/* two moons in two dimensions: a task no line settles */
	fn
	moons() -> Mat
	{
		let mut m = Mat { n: 0, d: 2, k: 2, x: Vec::new(), y: Vec::new(), c: Vec::new() };
		for i in 0..600{
			let t = (i % 300) as f64 / 300.0 * std::f64::consts::PI;
			let (a, b) = ((t.cos() * 3.0), (t.sin() * 3.0));
			let c = usize::from(i >= 300);
			m.x.extend_from_slice(&[a + 3.0 * c as f64, b + 1.5 - 3.0 * c as f64]);
			m.c.push(c);
			/* the target bends with the first feature */
			m.y.push(a * b / 10.0);
			m.n += 1;
		}
		m
	}

	#[test]
	fn forest_learns()
	{
		let m = moons();
		let f = forest(&[1.0], &m);
		let wrong = (0..m.n).filter(|&i| f.predict(m.row(i)) as usize != m.c[i]).count();
		assert!(wrong <= 6, "{} rows wrong", wrong);
		let mut reg = moons();
		reg.k = 0;
		let f = forest(&[1.0], &reg);
		let err = (0..reg.n).map(|i| (f.predict(reg.row(i)) - reg.y[i]).abs()).sum::<f64>() / reg.n as f64;
		assert!(err < 0.05, "mean error {}", err);
		/* the same seed, the same forest */
		let g = forest(&[1.0], &reg);
		for i in 0..reg.n{
			close(f.predict(reg.row(i)), g.predict(reg.row(i)));
		}
	}

	#[test]
	fn boosting_learns()
	{
		let m = moons();
		let b = boost(&[3.0], &m);
		let wrong = (0..m.n).filter(|&i| b.predict(m.row(i)) as usize != m.c[i]).count();
		assert!(wrong <= 12, "{} rows wrong", wrong);
		/* the target lies well away from zero: the start at the mean must be counted once */
		let mut reg = moons();
		reg.k = 0;
		for y in &mut reg.y{
			*y = 2.0 + *y;
		}
		let b = boost(&[3.0], &reg);
		let err = (0..reg.n).map(|i| (b.predict(reg.row(i)) - reg.y[i]).abs()).sum::<f64>() / reg.n as f64;
		let flat = mean(&reg.y);
		let base = reg.y.iter().map(|y| (y - flat).abs()).sum::<f64>() / reg.n as f64;
		assert!(err < base / 4.0, "mean error {} against {}", err, base);
	}
}
