/* feat: what a formula looks like, before anyone tries to solve it */
use std::io::Read;
use crate::sexp::{Sexp, Tok};

/* the operators and binders counted, one column each; kept sorted */
pub static OPS: &[&str] = &[
	"=", "=>", "and",
	"bvadd", "bvand", "bvashr", "bvcomp", "bvlshr", "bvmul", "bvnand", "bvneg", "bvnor",
	"bvnot", "bvor", "bvredand", "bvredor", "bvredxor", "bvsdiv", "bvsge", "bvsgt", "bvshl",
	"bvsle", "bvslt", "bvsmod", "bvsrem", "bvsub", "bvudiv", "bvuge", "bvugt", "bvule", "bvult",
	"bvurem", "bvxnor", "bvxor",
	"concat", "distinct", "exists", "extract", "forall", "ite", "let", "not", "or", "repeat",
	"rotate-left", "rotate-right", "rotate_left", "rotate_right", "select", "sign_extend",
	"store", "xor", "zero_extend",
];

/* the multiplicative ones, whose bit-blasting is the dearest */
static MULS: [&str; 6] = ["bvmul", "bvsdiv", "bvsmod", "bvsrem", "bvudiv", "bvurem"];

/* the index of an operator in OPS */
pub fn
op(name: &[u8]) -> Option<usize>
{
	OPS.binary_search_by(|o| o.as_bytes().cmp(name)).ok()
}

#[derive(Clone, Copy, PartialEq)]
pub enum Cmd {
	Assert,
	CheckSat,
	DeclareConst,
	DeclareFun,
	DefineFun,
	DefineFunRec,
	DefineFunsRec,
	SetInfo,
	SetLogic,
	Other,
}

/* the commands of the smt-lib language, with those the scanner tells apart */
static CMDS: [(&str, Cmd); 30] = [
	("assert", Cmd::Assert),
	("check-sat", Cmd::CheckSat),
	("check-sat-assuming", Cmd::Other),
	("declare-const", Cmd::DeclareConst),
	("declare-datatype", Cmd::Other),
	("declare-datatypes", Cmd::Other),
	("declare-fun", Cmd::DeclareFun),
	("declare-sort", Cmd::Other),
	("define-fun", Cmd::DefineFun),
	("define-fun-rec", Cmd::DefineFunRec),
	("define-funs-rec", Cmd::DefineFunsRec),
	("define-sort", Cmd::Other),
	("echo", Cmd::Other),
	("exit", Cmd::Other),
	("get-assertions", Cmd::Other),
	("get-assignment", Cmd::Other),
	("get-info", Cmd::Other),
	("get-model", Cmd::Other),
	("get-option", Cmd::Other),
	("get-proof", Cmd::Other),
	("get-unsat-assumptions", Cmd::Other),
	("get-unsat-core", Cmd::Other),
	("get-value", Cmd::Other),
	("pop", Cmd::Other),
	("push", Cmd::Other),
	("reset", Cmd::Other),
	("reset-assertions", Cmd::Other),
	("set-info", Cmd::SetInfo),
	("set-logic", Cmd::SetLogic),
	("set-option", Cmd::Other),
];

fn
cmd(name: &[u8]) -> Option<Cmd>
{
	CMDS.binary_search_by(|c| c.0.as_bytes().cmp(name)).ok().map(|i| CMDS[i].1)
}

#[derive(Clone, Copy, PartialEq)]
enum Head {
	None,
	/* (_ ...): an indexed identifier, named by the atom that follows */
	Idx,
	Cmd(Cmd),
	Op,
	Let,
	/* the binding list of a let, and one binding in it */
	Binds,
	Bind,
	Array,
	Bitvec,
	Bvlit,
	Other,
}

struct Frame {
	head: Head,
	nchild: usize,
	/* nesting of this list within a term; NOTERM when it lies outside every term */
	dep: usize,
	/* what the head makes of it: a width, or a flag */
	val: u64,
}

const NOTERM: usize = usize::MAX;

fn
decl(h: Head) -> bool
{
	matches!(h, Head::Cmd(Cmd::DeclareFun | Cmd::DeclareConst))
}

/* a term begins under assert, and as the body of a definition */
fn
termstart(h: Head, n: usize) -> bool
{
	matches!(h, Head::Cmd(Cmd::Assert)) && n == 1
		|| matches!(h, Head::Cmd(Cmd::DefineFun | Cmd::DefineFunRec)) && n == 4
}

fn
numeral(a: &[u8]) -> u64
{
	match std::str::from_utf8(a).ok().and_then(|s| s.parse().ok()){
		Some(n) => n,
		None => 0,
	}
}

fn
text(a: &[u8]) -> String
{
	String::from_utf8_lossy(a).into_owned()
}

pub struct Feat {
	pub id: String,
	pub logic: String,
	pub status: String,
	pub bytes: u64,
	pub asserts: u64,
	pub decls: u64,
	pub defs: u64,
	pub checks: u64,
	/* the deepest nesting of lists in an asserted term or a definition body */
	pub depth: usize,
	pub other: u64,
	pub bvlits: u64,
	pub arrays: u64,
	pub bits: u64,
	pub wmax: u64,
	pub ops: Vec<u64>,
	widths: Vec<u64>,
	stack: Vec<Frame>,
	/* within the sort of a declared symbol */
	indecl: bool,
	/* heads met in terms that OPS does not know, with their counts */
	pub unknown: Vec<(Vec<u8>, u64)>,
}

impl Feat {
	pub fn
	new(id: &str, logic: &str) -> Feat
	{
		Feat {
			id: id.to_string(),
			logic: logic.to_string(),
			status: String::new(),
			bytes: 0,
			asserts: 0,
			decls: 0,
			defs: 0,
			checks: 0,
			depth: 0,
			other: 0,
			bvlits: 0,
			arrays: 0,
			bits: 0,
			wmax: 0,
			ops: vec![0; OPS.len()],
			widths: Vec::new(),
			stack: Vec::new(),
			indecl: false,
			unknown: Vec::new(),
		}
	}

	/* applications of every operator */
	pub fn
	nops(&self) -> u64
	{
		self.ops.iter().sum()
	}

	/* the share of multiplicative and division operators among all operators */
	pub fn
	mulrat(&self) -> f64
	{
		let n = self.nops();
		if(n == 0){
			return 0.0;
		}
		let mut m = 0;
		for name in MULS{
			m += self.ops[op(name.as_bytes()).unwrap_or(0)];
		}
		m as f64 / n as f64
	}

	/* the number of distinct bitvector widths among the declared variables */
	pub fn
	wdist(&self) -> usize
	{
		self.widths.len()
	}

	pub fn
	header() -> String
	{
		let mut h = String::from("id\tlogic\tstatus\tbytes\tasserts\tdecls\tdefs\tchecks\tdepth\tnops\tmulrat\tbits\twmax\twdist\tarrays\tbvlits\tother");
		for o in OPS{
			h.push('\t');
			h.push_str(o);
		}
		h
	}

	pub fn
	row(&self) -> String
	{
		let mut s = format!(
			"{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{}\t{}\t{}\t{}\t{}",
			self.id, self.logic, self.status, self.bytes, self.asserts, self.decls, self.defs,
			self.checks, self.depth, self.nops(), self.mulrat(), self.bits, self.wmax,
			self.wdist(), self.arrays, self.bvlits, self.other);
		for n in &self.ops{
			s.push('\t');
			s.push_str(&n.to_string());
		}
		s
	}

	pub fn
	scan<R: Read>(&mut self, sx: &mut Sexp<R>) -> std::io::Result<()>
	{
		loop {
			match(sx.tok()?){
				Tok::Open => self.open(),
				Tok::Close => self.close(),
				Tok::Atom(a) => self.atom(a),
				Tok::Eof => return Ok(()),
			}
		}
	}

	/* a list opens; n counts the children its parent has so far, the head among them */
	fn
	open(&mut self)
	{
		let mut head = Head::None;
		let mut dep = NOTERM;
		if let Some(p) = self.stack.last_mut(){
			let n = p.nchild;
			p.nchild += 1;
			/* the list of bodies in a define-funs-rec holds terms, and is none itself */
			let bodies = matches!(p.head, Head::Cmd(Cmd::DefineFunsRec)) && n == 2;
			dep = if(bodies){
				0
			}else if(termstart(p.head, n)){
				1
			}else if(p.dep == NOTERM){
				NOTERM
			}else{
				p.dep + 1
			};
			head = match(p.head){
				Head::Let if n == 1 => Head::Binds,
				Head::Binds => Head::Bind,
				_ => Head::None,
			};
			if(decl(p.head) && n == 2){
				self.indecl = true;
			}
		}
		self.stack.push(Frame { head, nchild: 0, dep, val: 0 });
		if(dep != NOTERM && dep > self.depth){
			self.depth = dep;
		}
	}

	fn
	close(&mut self)
	{
		let Some(f) = self.stack.pop() else {
			return;
		};
		match(f.head){
			Head::Bitvec => {
				/* the sort of a declared variable, as in (declare-fun x () (_ BitVec 32)) */
				if(f.val > 0 && self.indecl && self.stack.last().map(|p| decl(p.head)) == Some(true)){
					self.bits += f.val;
					if(f.val > self.wmax){
						self.wmax = f.val;
					}
					if !self.widths.contains(&f.val){
						self.widths.push(f.val);
					}
				}
			}
			Head::Cmd(Cmd::DeclareFun | Cmd::DeclareConst) => {
				self.decls += 1;
				self.indecl = false;
			}
			Head::Cmd(Cmd::DefineFun | Cmd::DefineFunRec | Cmd::DefineFunsRec) => self.defs += 1,
			_ => {}
		}
	}

	/* an atom arrives; n counts the children its list has so far, the head among them */
	fn
	atom(&mut self, a: &[u8])
	{
		if(a.len() > 1 && a[0] == b'#' && (a[1] == b'x' || a[1] == b'b')){
			self.bvlits += 1;
		}
		if(self.stack.is_empty()){
			return;
		}
		let k = self.stack.len() - 1;
		let f = &mut self.stack[k];
		let n = f.nchild;
		f.nchild += 1;
		match(f.head){
			Head::None => {
				if(n == 0){
					self.sethead(k, a);
				}
			}
			Head::Idx => {
				if(n == 1){
					self.indexed(k, a);
				}
			}
			/* the width follows the _, as in (_ BitVec 32) */
			Head::Bitvec => {
				if(n == 2){
					self.stack[k].val = numeral(a);
				}
			}
			Head::Cmd(c) => self.cmdatom(k, c, n, a),
			_ => {}
		}
	}

	fn
	sethead(&mut self, k: usize, a: &[u8])
	{
		/* the head of a let binding is the variable bound, not an application */
		if(self.stack[k].head == Head::Bind){
			return;
		}
		if(k == 0){
			if let Some(c) = cmd(a){
				match(c){
					Cmd::Assert => self.asserts += 1,
					Cmd::CheckSat => self.checks += 1,
					_ => {}
				}
				self.stack[k].head = Head::Cmd(c);
				return;
			}
		}
		match(a){
			b"_" => self.stack[k].head = Head::Idx,
			b"Array" => {
				self.stack[k].head = Head::Array;
				/* the sort of a declared variable, as in (declare-fun m () (Array ...)) */
				if(self.indecl && k > 0 && decl(self.stack[k - 1].head)){
					self.arrays += 1;
				}
			}
			_ => match op(a){
				Some(i) => {
					self.ops[i] += 1;
					self.stack[k].head = if(OPS[i] == "let"){ Head::Let } else { Head::Op };
				}
				None => self.forgot(k, a),
			},
		}
	}

	/* the name inside an indexed identifier, as the extract of (_ extract 7 0) */
	fn
	indexed(&mut self, k: usize, a: &[u8])
	{
		if(a == b"BitVec"){
			self.stack[k].head = Head::Bitvec;
		}else if(a.len() > 2 && a.starts_with(b"bv") && a[2..].iter().all(|c| c.is_ascii_digit())){
			self.bvlits += 1;
			self.stack[k].head = Head::Bvlit;
		}else if let Some(i) = op(a){
			self.ops[i] += 1;
			self.stack[k].head = Head::Op;
		}else{
			self.forgot(k, a);
		}
	}

	/* a head OPS does not know: a macro of the benchmark, or an operator left out */
	fn
	forgot(&mut self, k: usize, a: &[u8])
	{
		self.stack[k].head = Head::Other;
		if(self.stack[k].dep != NOTERM){
			self.other += 1;
			match self.unknown.iter_mut().find(|u| u.0 == a){
				Some(u) => u.1 += 1,
				None => self.unknown.push((a.to_vec(), 1)),
			}
		}
	}

	fn
	cmdatom(&mut self, k: usize, c: Cmd, n: usize, a: &[u8])
	{
		match(c){
			Cmd::SetLogic => {
				if(n == 1){
					self.logic = text(a);
				}
			}
			Cmd::SetInfo => {
				if(n == 1){
					self.stack[k].val = u64::from(a == b":status");
				}else if(n == 2 && self.stack[k].val == 1){
					self.status = text(a);
				}
			}
			/* the sort of the declared symbol, when it is a plain name */
			Cmd::DeclareFun | Cmd::DeclareConst => {
				if(n == 2){
					self.indecl = true;
				}
			}
			_ => {}
		}
	}
}

#[cfg(test)]
mod test {
	use super::*;
	use crate::buf::Buf;

	fn
	feats(s: &str) -> Feat
	{
		let mut sx = Sexp::new(Buf::new(s.as_bytes()));
		let mut f = Feat::new("t", "");
		f.scan(&mut sx).unwrap();
		f.bytes = sx.tell();
		f
	}

	#[test]
	fn tables_are_sorted()
	{
		for i in 1..OPS.len(){
			assert!(OPS[i - 1] < OPS[i], "{} {}", OPS[i - 1], OPS[i]);
		}
		for i in 1..CMDS.len(){
			assert!(CMDS[i - 1].0 < CMDS[i].0, "{} {}", CMDS[i - 1].0, CMDS[i].0);
		}
		for name in MULS.into_iter().chain(["extract", "let", "forall", "exists"]){
			assert!(op(name.as_bytes()).is_some(), "{}", name);
		}
	}

	#[test]
	fn one_formula()
	{
		let s = r#"; a comment with (parentheses) and bvadd
(set-logic QF_ABV)
(set-info :source "Generated by: (a test)")
(set-info :status unsat)
(set-option :produce-models true)
(declare-fun x () (_ BitVec 32))
(declare-const y (_ BitVec 8))
(declare-fun a ((Array (_ BitVec 4) (_ BitVec 16))) Bool)
(declare-fun m (Array (_ BitVec 4) (_ BitVec 16)))
(define-fun z () (_ BitVec 64) (bvmul (concat x y) #x0000000000000001))
(assert (! (= z (bvadd (select m #b0000) ((_ extract 15 0) x))) :named n1))
(assert (let ((t (bvnot x))) (and (bvult t (_ bv0 32)) t (distinct x y))))
(assert (ite true false (= y ((_ zero_extend 8) x))))
(check-sat)
(exit
"#;
		let f = feats(s);
		assert_eq!(f.logic, "QF_ABV");
		assert_eq!(f.status, "unsat");
		assert_eq!(f.bytes, s.len() as u64);
		assert_eq!(f.asserts, 3);
		assert_eq!(f.checks, 1);
		assert_eq!(f.decls, 4);
		assert_eq!(f.defs, 1);
		/* in the first assert, the extract application lies under the annotation, =,
		 * bvadd and select: five lists */
		assert_eq!(f.depth, 5);
		assert_eq!(f.arrays, 1);
		assert_eq!(f.bits, 40);
		assert_eq!(f.wmax, 32);
		assert_eq!(f.wdist(), 2);
		assert_eq!(f.bvlits, 3);
		assert_eq!(f.nops(), 14);
		assert_eq!(f.mulrat(), 1.0 / 14.0);
		assert_eq!(f.other, 1);
		assert_eq!(f.ops[op(b"select").unwrap()], 1);
		assert_eq!(f.ops[op(b"extract").unwrap()], 1);
		assert_eq!(f.ops[op(b"zero_extend").unwrap()], 1);
		assert_eq!(f.ops[op(b"let").unwrap()], 1);
		assert_eq!(f.unknown, [(b"!".to_vec(), 1)]);
	}

	#[test]
	fn term_depth()
	{
		/* the catalog gives (a (b (c d) (e (f g)))) four: it counts the lists */
		assert_eq!(feats("(assert (= (a (b (c d) (e (f g)))) h))").depth, 5);
		assert_eq!(feats("(assert x)").depth, 0);
		assert_eq!(feats("(assert (bvadd x y))").depth, 1);
		assert_eq!(feats("(assert (bvadd (_ extract 7 0) y))").depth, 2);
		/* binding lists count with the rest */
		assert_eq!(feats("(assert (let ((x (f (g y)))) x))").depth, 5);
		assert_eq!(feats("(assert (forall ((x Bool)) (p x)))").depth, 3);
		/* outside assert, only the bodies of definitions are terms */
		assert_eq!(feats("(define-fun f ((x Bool)) Bool (and x true))").depth, 1);
		assert_eq!(feats("(define-funs-rec ((f ((x Bool)) Bool)) ((not x)))").depth, 1);
		assert_eq!(feats("(get-value ((f x)))").depth, 0);
	}

	#[test]
	fn macros_are_not_operators()
	{
		/* the head of a binding names a variable; a macro called in a term is counted apart */
		let f = feats("(assert (let ((helper (f x))) (helper y)))");
		assert_eq!(f.other, 2);
		assert_eq!(f.unknown, [(b"f".to_vec(), 1), (b"helper".to_vec(), 1)]);
		assert_eq!(f.ops[op(b"let").unwrap()], 1);
		/* and an unknown indexed identifier is told apart too */
		let f = feats("(assert ((_ left_out 3) x))");
		assert_eq!(f.unknown, [(b"left_out".to_vec(), 1)]);
	}

	#[test]
	fn streams_that_end_early()
	{
		let f = feats("(set-logic QF_BV)(declare-fun x () (_ BitVec 8))(assert (bvadd x");
		assert_eq!(f.logic, "QF_BV");
		assert_eq!(f.decls, 1);
		assert_eq!(f.bits, 8);
		assert_eq!(f.asserts, 1);
		assert_eq!(f.depth, 1);
	}
}
