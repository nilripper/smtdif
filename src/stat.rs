/* stat: plain statistics, of a sample and of a prediction */
use std::f64::consts::PI;

pub fn
mean(v: &[f64]) -> f64
{
	if(v.is_empty()){
		return 0.0;
	}
	v.iter().sum::<f64>() / v.len() as f64
}

/* the sample standard deviation */
pub fn
sd(v: &[f64]) -> f64
{
	if(v.len() < 2){
		return 0.0;
	}
	let m = mean(v);
	(v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

/* the q quantile of a sorted sample, interpolated between its order statistics */
pub fn
quantile(sorted: &[f64], q: f64) -> f64
{
	if(sorted.is_empty()){
		return 0.0;
	}
	let pos = q * (sorted.len() - 1) as f64;
	let lo = pos.floor() as usize;
	let hi = pos.ceil() as usize;
	sorted[lo] + (pos - lo as f64) * (sorted[hi] - sorted[lo])
}

pub fn
median(v: &[f64]) -> f64
{
	let mut s = v.to_vec();
	s.sort_by(|a, b| a.total_cmp(b));
	quantile(&s, 0.5)
}

/* the ranks of v from 1 to n, equal values sharing their mean */
pub fn
rank(v: &[f64]) -> Vec<f64>
{
	let mut ord: Vec<usize> = (0..v.len()).collect();
	ord.sort_by(|a, b| v[*a].total_cmp(&v[*b]));
	let mut r = vec![0.0; v.len()];
	let mut i = 0;
	while(i < ord.len()){
		let mut j = i;
		/* by total order, so that a value gone astray still ends its run */
		while(j < ord.len() && v[ord[j]].total_cmp(&v[ord[i]]) == std::cmp::Ordering::Equal){
			j += 1;
		}
		for k in i..j{
			r[ord[k]] = (i + j + 1) as f64 / 2.0;
		}
		i = j;
	}
	r
}

/* pearson's correlation; 0 when either side stands still */
pub fn
pearson(x: &[f64], y: &[f64]) -> f64
{
	let (mx, my) = (mean(x), mean(y));
	let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
	for i in 0..x.len(){
		let (a, b) = (x[i] - mx, y[i] - my);
		sxy += a * b;
		sxx += a * a;
		syy += b * b;
	}
	if(sxx == 0.0 || syy == 0.0){
		return 0.0;
	}
	sxy / (sxx * syy).sqrt()
}

/* spearman's correlation of posts: pearson over the ranks */
pub fn
spearman(x: &[f64], y: &[f64]) -> f64
{
	pearson(&rank(x), &rank(y))
}

pub fn
mae(y: &[f64], p: &[f64]) -> f64
{
	if(y.is_empty()){
		return 0.0;
	}
	y.iter().zip(p).map(|(a, b)| (a - b).abs()).sum::<f64>() / y.len() as f64
}

pub fn
rmse(y: &[f64], p: &[f64]) -> f64
{
	if(y.is_empty()){
		return 0.0;
	}
	(y.iter().zip(p).map(|(a, b)| (a - b) * (a - b)).sum::<f64>() / y.len() as f64).sqrt()
}

/* the proportion of the variance of y that p accounts for */
pub fn
r2(y: &[f64], p: &[f64]) -> f64
{
	let s = y.iter().map(|a| (a - mean(y)) * (a - mean(y))).sum::<f64>();
	if(s == 0.0){
		return 0.0;
	}
	1.0 - y.iter().zip(p).map(|(a, b)| (a - b) * (a - b)).sum::<f64>() / s
}

/* a confusion matrix over k classes, by truth and by prediction */
#[derive(Clone, Default)]
pub struct Conf {
	pub k: usize,
	n: Vec<usize>,
}

impl Conf {
	pub fn
	new(k: usize) -> Conf
	{
		Conf { k, n: vec![0; k * k] }
	}

	pub fn
	add(&mut self, truth: usize, pred: usize)
	{
		self.n[truth * self.k + pred] += 1;
	}

	pub fn
	get(&self, truth: usize, pred: usize) -> usize
	{
		self.n[truth * self.k + pred]
	}

	pub fn
	total(&self) -> usize
	{
		self.n.iter().sum()
	}

	pub fn
	accuracy(&self) -> f64
	{
		let t = self.total();
		if(t == 0){
			return 0.0;
		}
		(0..self.k).map(|c| self.n[c * self.k + c]).sum::<usize>() as f64 / t as f64
	}

	/* of those called c, the share that are c */
	pub fn
	precision(&self, c: usize) -> f64
	{
		let called: usize = (0..self.k).map(|t| self.n[t * self.k + c]).sum();
		if(called == 0){
			return 0.0;
		}
		self.n[c * self.k + c] as f64 / called as f64
	}

	/* of those that are c, the share called c */
	pub fn
	recall(&self, c: usize) -> f64
	{
		let real: usize = self.n[c * self.k..c * self.k + self.k].iter().sum();
		if(real == 0){
			return 0.0;
		}
		self.n[c * self.k + c] as f64 / real as f64
	}

	pub fn
	f1(&self, c: usize) -> f64
	{
		let (p, r) = (self.precision(c), self.recall(c));
		if(p + r == 0.0){
			return 0.0;
		}
		2.0 * p * r / (p + r)
	}

	/* the mean f1 over the classes the truth holds */
	pub fn
	macro_f1(&self) -> f64
	{
		let (mut s, mut m) = (0.0, 0);
		for c in 0..self.k{
			if(self.n[c * self.k..c * self.k + self.k].iter().any(|n| *n > 0)){
				s += self.f1(c);
				m += 1;
			}
		}
		if(m == 0){
			return 0.0;
		}
		s / m as f64
	}
}

/* the logistic function, kept from overflowing */
pub fn
logistic(z: f64) -> f64
{
	if(z >= 0.0){
		1.0 / (1.0 + (-z).exp())
	}else{
		let e = z.exp();
		e / (1.0 + e)
	}
}

/* the density of the standard normal */
pub fn
normal_pdf(z: f64) -> f64
{
	(-z * z / 2.0).exp() / (2.0 * PI).sqrt()
}

/* where the largest score stands, the first of equals: a float has no total order of its own */
pub fn
argmax(score: &[f64]) -> usize
{
	(1..score.len()).fold(0, |b, i| if(score[i] > score[b]){ i } else { b })
}

#[cfg(test)]
mod test {
	use super::*;

	fn
	close(a: f64, b: f64)
	{
		assert!((a - b).abs() < 1e-9, "{} is not {}", a, b);
	}

	#[test]
	fn moments()
	{
		close(mean(&[1.0, 2.0, 3.0, 4.0]), 2.5);
		close(sd(&[1.0, 2.0, 3.0, 4.0]), (5.0f64 / 3.0).sqrt());
		let s = [1.0, 2.0, 3.0, 4.0];
		close(quantile(&s, 0.0), 1.0);
		close(quantile(&s, 0.25), 1.75);
		close(quantile(&s, 0.5), 2.5);
		close(quantile(&s, 1.0), 4.0);
		close(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
		close(sd(&[7.0]), 0.0);
	}

	#[test]
	fn ranks_and_correlations()
	{
		let r = rank(&[3.0, 1.0, 2.0, 2.0]);
		for (a, b) in r.iter().zip([4.0, 1.0, 2.5, 2.5]){
			close(*a, b);
		}
		let x = [1.0, 2.0, 3.0, 4.0, 5.0];
		let y = [2.0, 1.0, 4.0, 3.0, 5.0];
		/* with no ties the ranks are the values up to a line, so both agree */
		close(spearman(&x, &y), 0.8);
		close(pearson(&x, &y), 0.8);
		close(spearman(&x, &[7.0, 7.0, 7.0, 7.0, 7.0]), 0.0);
		close(spearman(&x, &[5.0, 4.0, 3.0, 2.0, 1.0]), -1.0);
	}

	#[test]
	fn errors()
	{
		let y = [1.0, 2.0, 3.0];
		close(mae(&y, &[2.0, 2.0, 1.0]), 1.0);
		close(rmse(&y, &[2.0, 2.0, 1.0]), (5.0f64 / 3.0).sqrt());
		close(r2(&y, &[2.0, 2.0, 1.0]), -1.5);
		close(r2(&y, &[1.0, 2.0, 3.0]), 1.0);
	}

	#[test]
	fn confusion()
	{
		let mut c = Conf::new(3);
		for (t, p) in [(0, 0), (0, 0), (0, 1), (1, 1), (1, 2), (2, 0)]{
			c.add(t, p);
		}
		close(c.accuracy(), 0.5);
		close(c.precision(0), 2.0 / 3.0);
		close(c.recall(0), 2.0 / 3.0);
		close(c.f1(0), 2.0 / 3.0);
		close(c.f1(1), 0.5);
		close(c.f1(2), 0.0);
		close(c.macro_f1(), (2.0 / 3.0 + 0.5) / 3.0);
		close(Conf::new(2).macro_f1(), 0.0);
	}

	#[test]
	fn functions()
	{
		close(logistic(0.0), 0.5);
		close(logistic(1000.0), 1.0);
		close(logistic(-1000.0), 0.0);
		close(normal_pdf(0.0), 1.0 / (2.0 * PI).sqrt());
	}
}
