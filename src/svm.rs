/* svm: a support vector machine over a radial kernel, by dual coordinate descent.
 * the kernel holds a constant, k + 1, so that the bias is regularized with the rest
 * (hsieh, chang and lin 2008 for the classes; ho and lin 2012 for the regression) */
use crate::model::{Mat, Model};
use crate::rng::Rand;
use crate::stat::argmax;

/* the fitting ends once no weight moves further than this share of the cap in a sweep */
static TOL: f64 = 1e-3;
static SWEEPS: usize = 300;

struct Svm {
	d: usize,
	gamma: f64,
	/* one fit against the rest for each class; 0 when the machine regresses */
	k: usize,
	/* the support vectors, row after row, and their weights: one a class, or the one */
	sv: Vec<f64>,
	coef: Vec<f64>,
}

impl Svm {
	fn
	kern(&self, x: &[f64], i: usize) -> f64
	{
		let v = &self.sv[i * self.d..i * self.d + self.d];
		let d = v.iter().zip(x).map(|(a, b)| (a - b) * (a - b)).sum::<f64>();
		(-self.gamma * d).exp() + 1.0
	}
}

impl Model for Svm {
	fn
	predict(&self, x: &[f64]) -> f64
	{
		let n = self.sv.len() / self.d;
		if(self.k == 0){
			return (0..n).map(|i| self.coef[i] * self.kern(x, i)).sum();
		}
		let score: Vec<f64> = (0..self.k).map(|c| (0..n).map(|i| self.coef[c * n + i] * self.kern(x, i)).sum::<f64>()).collect();
		argmax(&score) as f64
	}
}

/* the kernel over every pair of the training rows */
fn
gram(m: &Mat, gamma: f64) -> Vec<f64>
{
	let n = m.n;
	let mut g = vec![2.0; n * n];
	for i in 0..n{
		for j in i + 1..n{
			let d = m.row(i).iter().zip(m.row(j)).map(|(a, b)| (a - b) * (a - b)).sum::<f64>();
			let k = (-gamma * d).exp() + 1.0;
			g[i * n + j] = k;
			g[j * n + i] = k;
		}
	}
	g
}

/* one sweep of coordinate descent over the rows, in a fresh order; tells how far a weight moved.
 * a row weighs u[i] between lo and hi, and enters the kernel signed by z[i]: the gradient
 * along its weight is z[i] gu[i] - t[i], with a dead zone of eps about the target */
fn
sweep(g: &[f64], z: &[f64], t: &[f64], eps: f64, lo: f64, hi: f64,
	u: &mut [f64], gu: &mut [f64], order: &mut [usize], rand: &mut Rand) -> f64
{
	let n = u.len();
	rand.shuffle(order);
	let mut most = 0.0f64;
	for i in order.iter(){
		let i = *i;
		let h = z[i] * gu[i] - t[i] - g[i * n + i] * u[i];
		let want = if(h > eps){
			-(h - eps) / g[i * n + i]
		}else if(h < -eps){
			-(h + eps) / g[i * n + i]
		}else{
			0.0
		};
		let un = want.clamp(lo, hi);
		let step = z[i] * (un - u[i]);
		if(step != 0.0){
			most = most.max((un - u[i]).abs());
			u[i] = un;
			for (a, k) in gu.iter_mut().zip(&g[i * n..i * n + n]){
				*a += step * k;
			}
		}
	}
	most
}

/* the signed weight of every training row, from one dual fit */
fn
dual(g: &[f64], z: &[f64], t: &[f64], eps: f64, lo: f64, hi: f64, seed: u64) -> Vec<f64>
{
	let n = t.len();
	let (mut u, mut gu) = (vec![0.0; n], vec![0.0; n]);
	let mut order: Vec<usize> = (0..n).collect();
	let mut rand = Rand::new(seed);
	for _ in 0..SWEEPS{
		if(sweep(g, z, t, eps, lo, hi, &mut u, &mut gu, &mut order, &mut rand) <= TOL * hi){
			break;
		}
	}
	u
}

pub fn
svm(p: &[f64], m: &Mat) -> Box<dyn Model>
{
	let (cap, gamma) = (p[0], p[1] / m.d as f64);
	let n = m.n;
	let g = gram(m, gamma);
	/* the signed weights of the rows: one set in a regression, one a class in a classification */
	let mut sets: Vec<Vec<f64>> = Vec::new();
	if(m.k == 0){
		let u = dual(&g, &vec![1.0; n], &m.y, p[2], -cap, cap, 0);
		sets.push(u);
	}else{
		for c in 0..m.k{
			let z: Vec<f64> = m.c.iter().map(|k| if(*k == c){ 1.0 } else { -1.0 }).collect();
			let u = dual(&g, &z, &vec![1.0; n], 0.0, 0.0, cap, c as u64);
			sets.push(u.iter().zip(&z).map(|(a, z)| a * z).collect());
		}
	}
	drop(g);
	/* the support vectors: the rows that carry a weight */
	let (mut sv, mut coef) = (Vec::new(), vec![Vec::new(); sets.len()]);
	for i in 0..n{
		if(sets.iter().any(|a| a[i] != 0.0)){
			sv.extend_from_slice(m.row(i));
			for s in 0..sets.len(){
				coef[s].push(sets[s][i]);
			}
		}
	}
	Box::new(Svm { d: m.d, gamma, k: m.k, sv, coef: coef.concat() })
}

#[cfg(test)]
mod test {
	use super::*;

	fn
	close(a: f64, b: f64, tol: f64)
	{
		assert!((a - b).abs() < tol, "{} is not {}", a, b);
	}

	/* n rows on a line, each at its own place, with the target f */
	fn
	line(n: usize, f: impl Fn(f64) -> f64) -> Mat
	{
		let mut m = Mat { n, d: 1, k: 0, x: Vec::new(), y: Vec::new(), c: Vec::new() };
		for i in 0..n{
			let x = -3.0 + 6.0 * i as f64 / (n - 1) as f64;
			m.x.push(x);
			m.y.push(f(x));
			m.c.push(usize::from(x > 0.0));
		}
		m
	}

	#[test]
	fn classes()
	{
		let mut m = line(100, |x| x);
		m.k = 2;
		let s = svm(&[10.0, 2.0], &m);
		for i in 0..m.n{
			assert_eq!(s.predict(m.row(i)) as usize, m.c[i], "row {}", i);
		}
		assert_eq!(s.predict(&[-1.0]), 0.0);
		assert_eq!(s.predict(&[1.0]), 1.0);
		/* a third class on the far side */
		m.k = 3;
		for i in 0..m.n{
			if(m.x[i] > 2.0){
				m.c[i] = 2;
			}
		}
		let s = svm(&[10.0, 2.0], &m);
		for i in 0..m.n{
			assert_eq!(s.predict(m.row(i)) as usize, m.c[i], "row {}", i);
		}
	}

	#[test]
	fn regression()
	{
		let m = line(200, f64::sin);
		let s = svm(&[100.0, 2.0, 0.05], &m);
		let err = (0..m.n).map(|i| (s.predict(m.row(i)) - m.y[i]).abs()).sum::<f64>() / m.n as f64;
		assert!(err < 0.04, "mean error {}", err);
		close(s.predict(&[1.0]), 1.0f64.sin(), 0.1);
		close(s.predict(&[-2.0]), (-2.0f64).sin(), 0.1);
	}

	#[test]
	fn weights_stay_within_the_cap()
	{
		let m = line(40, f64::sin);
		let s = svm(&[1.0, 1.0, 0.1], &m);
		/* every weight lies in [-1, 1], so the prediction lies within their number */
		for i in 0..m.n{
			assert!(s.predict(m.row(i)).abs() <= m.n as f64 + 1.0);
		}
	}
}
