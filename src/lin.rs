/* lin: ridge regression and logistic regression, over the same normal equations */
use crate::model::{Mat, Model};
use crate::stat::{argmax, logistic, mean};

/* a = l l', with l left in the lower triangle of a; false when a is not positive definite */
fn
cholesky(a: &mut [f64], d: usize) -> bool
{
	for j in 0..d{
		let mut s = a[j * d + j];
		for k in 0..j{
			s -= a[j * d + k] * a[j * d + k];
		}
		if !(s > 0.0){
			return false;
		}
		a[j * d + j] = s.sqrt();
		for i in j + 1..d{
			let mut s = a[i * d + j];
			for k in 0..j{
				s -= a[i * d + k] * a[j * d + k];
			}
			a[i * d + j] = s / a[j * d + j];
		}
	}
	true
}

/* l l' w = b, with w left in b */
fn
solve(l: &[f64], b: &mut [f64], d: usize)
{
	for i in 0..d{
		for j in 0..i{
			b[i] -= l[i * d + j] * b[j];
		}
		b[i] /= l[i * d + i];
	}
	for i in (0..d).rev(){
		for j in i + 1..d{
			b[i] -= l[j * d + i] * b[j];
		}
		b[i] /= l[i * d + i];
	}
}

/* ln(1 + e^t), in a form that holds for a large t */
fn
softplus(t: f64) -> f64
{
	if(t > 0.0){ t + (-t).exp().ln_1p() } else { t.exp().ln_1p() }
}

struct Line {
	w: Vec<f64>,
	/* the means of the features over the training rows */
	mu: Vec<f64>,
	b: f64,
}

impl Model for Line {
	fn
	predict(&self, x: &[f64]) -> f64
	{
		self.b + x.iter().zip(&self.w).zip(&self.mu).map(|((v, w), m)| (v - m) * w).sum::<f64>()
	}
}

/* the plane of least squared error over the training rows, with lambda holding its slope down */
pub fn
ridge(p: &[f64], m: &Mat) -> Box<dyn Model>
{
	let (n, d) = (m.n, m.d);
	let b = mean(&m.y);
	/* the features about their means: the slope then reads off the errors alone */
	let mu: Vec<f64> = (0..d).map(|j| (0..n).map(|i| m.x[i * d + j]).sum::<f64>() / n as f64).collect();
	let mut a = vec![0.0; d * d];
	let mut w = vec![0.0; d];
	for i in 0..n{
		let at = |j: usize| m.x[i * d + j] - mu[j];
		let e = m.y[i] - b;
		for j in 0..d{
			w[j] += at(j) * e;
			for l in j..d{
				a[j * d + l] += at(j) * at(l);
			}
		}
	}
	for j in 0..d{
		w[j] /= n as f64;
		for l in j..d{
			a[j * d + l] /= n as f64;
			a[l * d + j] = a[j * d + l];
		}
		a[j * d + j] += p[0];
	}
	/* should the equations fail, the mean alone is left to answer */
	if(cholesky(&mut a, d)){
		solve(&a, &mut w, d);
	}else{
		w.iter_mut().for_each(|v| *v = 0.0);
	}
	Box::new(Line { w, mu, b })
}

/* one plane for each class, of d slopes and a constant; the class with the highest score wins */
struct Planes {
	w: Vec<f64>,
	d: usize,
}

impl Planes {
	fn
	score(&self, c: usize, x: &[f64]) -> f64
	{
		let w = &self.w[c * (self.d + 1)..(c + 1) * (self.d + 1)];
		x.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() + w[self.d]
	}
}

impl Model for Planes {
	fn
	predict(&self, x: &[f64]) -> f64
	{
		let score: Vec<f64> = (0..self.w.len() / (self.d + 1)).map(|c| self.score(c, x)).collect();
		argmax(&score) as f64
	}
}

/* one class against the rest, by newton's method on the mean log loss:
 * lambda holds the slopes down, the constant goes free */
fn
one(m: &Mat, want: usize, lam: f64) -> Vec<f64>
{
	let (n, d) = (m.n, m.d + 1);
	let score = |w: &[f64], i: usize| m.row(i).iter().zip(w).map(|(a, b)| a * b).sum::<f64>() + w[m.d];
	let loss = |w: &[f64]| {
		let mut l = 0.0;
		for i in 0..n{
			let s = score(w, i);
			l += if(m.c[i] == want){ softplus(-s) } else { softplus(s) };
		}
		l / n as f64 + lam * w[..m.d].iter().map(|a| a * a).sum::<f64>() / 2.0
	};
	/* the constants start at the log odds of the class */
	let few = m.c.iter().filter(|c| **c == want).count().max(1);
	let mut w = vec![0.0; d];
	w[m.d] = (few as f64 / (n - few).max(1) as f64).ln();
	let mut old = loss(&w);
	for _ in 0..100{
		/* the gradient and the hessian of the mean log loss */
		let (mut g, mut h) = (vec![0.0; d], vec![0.0; d * d]);
		for i in 0..n{
			let pr = logistic(score(&w, i));
			let (r, q) = (pr - f64::from(m.c[i] == want), pr * (1.0 - pr));
			let at = |j: usize| if(j == m.d){ 1.0 } else { m.x[i * m.d + j] };
			for j in 0..d{
				g[j] += at(j) * r;
				for l in j..d{
					h[j * d + l] += at(j) * at(l) * q;
				}
			}
		}
		for j in 0..d{
			g[j] /= n as f64;
			for l in j..d{
				h[j * d + l] /= n as f64;
				h[l * d + j] = h[j * d + l];
			}
			h[j * d + j] += if(j == m.d){ 1e-8 } else { lam };
		}
		if(!cholesky(&mut h, d)){
			break;
		}
		/* newton's step: the descent direction, halved for as long as it sends the loss up */
		for v in &mut g{
			*v = -*v;
		}
		solve(&h, &mut g, d);
		let (mut t, mut went, mut taken) = (1.0, false, 0.0);
		while(t >= 1e-6){
			let new: Vec<f64> = w.iter().zip(&g).map(|(a, b)| a + t * b).collect();
			let l = loss(&new);
			if(l <= old){
				taken = g.iter().fold(0.0f64, |s, v| s.max(t * v.abs()));
				w = new;
				old = l;
				went = true;
				break;
			}
			t /= 2.0;
		}
		if(!went || taken < 1e-9){
			break;
		}
	}
	w
}

/* logistic regression, a fit against the rest for each class */
pub fn
logisticfit(p: &[f64], m: &Mat) -> Box<dyn Model>
{
	let mut w = Vec::new();
	for c in 0..m.k{
		w.extend(one(m, c, p[0]));
	}
	Box::new(Planes { w, d: m.d })
}

#[cfg(test)]
mod test {
	use super::*;

	fn
	close(a: f64, b: f64)
	{
		assert!((a - b).abs() < 1e-6, "{} is not {}", a, b);
	}

	#[test]
	fn linear_systems()
	{
		/* the matrix is [[4, 2], [2, 3]], which sends (1.5, 2) to (10, 9) */
		let mut a = vec![4.0, 2.0, 2.0, 3.0];
		assert!(cholesky(&mut a, 2));
		let mut b = vec![10.0, 9.0];
		solve(&a, &mut b, 2);
		close(b[0], 1.5);
		close(b[1], 2.0);
		assert!(!cholesky(&mut [1.0, 2.0, 2.0, 1.0], 2));
		close(softplus(0.0), 0.6931471805599453);
		assert!((softplus(800.0) - 800.0).abs() < 1e-9);
	}

	fn
	data(n: usize) -> Mat
	{
		let mut m = Mat { n, d: 3, k: 2, x: Vec::new(), y: Vec::new(), c: Vec::new() };
		for i in 0..n{
			let (a, b, c) = ((i % 17) as f64 - 8.0, (i % 5) as f64 - 2.0, (i % 7) as f64);
			m.x.extend_from_slice(&[a, b, c]);
			m.y.push(2.0 * a - b + c + 3.0);
			m.c.push(usize::from(a + b > 0.0));
		}
		m
	}

	#[test]
	fn ridge_finds_the_plane()
	{
		let m = data(200);
		let r = ridge(&[1e-8], &m);
		for i in 0..m.n{
			close(r.predict(m.row(i)), m.y[i]);
		}
		/* a feature that never varies leaves its slope alone */
		let mut flat = data(50);
		for i in 0..flat.n{
			flat.x[i * flat.d] = 1.0;
		}
		let r = ridge(&[0.0], &flat);
		assert!(r.predict(&[1.0, 0.0, 0.0]).is_finite());
	}

	#[test]
	fn logistic_separates()
	{
		let m = data(300);
		let r = logisticfit(&[1e-3], &m);
		for i in 0..m.n{
			assert_eq!(r.predict(m.row(i)) as usize, m.c[i], "row {}", i);
		}
		/* the same with a class no row holds: it never wins */
		let mut m = data(30);
		m.k = 3;
		let r = logisticfit(&[1e-3], &m);
		assert!(m.c.iter().all(|c| *c < 2));
		for i in 0..m.n{
			assert_eq!(r.predict(m.row(i)) as usize, m.c[i]);
		}
	}
}
