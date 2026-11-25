/* flags: argument scanning in the manner of ARGBEGIN */

pub struct Flags {
	prog: String,
	seen: Vec<(char, String)>,
	pub argv: Vec<String>,
}

impl Flags {
	pub fn
	new() -> Flags
	{
		let mut prog = "?".to_string();
		if let Some(a) = std::env::args_os().next(){
			prog = a.to_string_lossy().into_owned();
			if let Some(base) = prog.rsplit(['/', '\\']).next(){
				prog = base.strip_suffix(".exe").unwrap_or(base).to_string();
			}
		}
		Flags { prog, seen: Vec::new(), argv: Vec::new() }
	}

	pub fn
	name(&self) -> &str
	{
		&self.prog
	}

	/* scan the arguments; a letter in bools takes no value, and such letters may be run together */
	pub fn
	parse(&mut self, bools: &str)
	{
		let args: Vec<String> = std::env::args_os().skip(1)
			.map(|a| a.to_string_lossy().into_owned()).collect();
		let mut i = 0;
		while(i < args.len()){
			let a = &args[i];
			if(a == "--"){
				i += 1;
				break;
			}
			if(a.len() < 2 || !a.starts_with('-')){
				break;
			}
			let letters: Vec<char> = a.chars().skip(1).collect();
			let mut j = 0;
			while(j < letters.len()){
				let c = letters[j];
				if(bools.contains(c)){
					self.seen.push((c, String::new()));
					j += 1;
					continue;
				}
				/* the value: what is left of this argument, or else the next one */
				let mut v: String = letters[j + 1..].iter().collect();
				if(v.starts_with('=')){
					v = v.chars().skip(1).collect();
				}
				if(v.len() == 0){
					i += 1;
					v = args.get(i).cloned().unwrap_or_default();
				}
				self.seen.push((c, v));
				j = letters.len();
			}
			i += 1;
		}
		self.argv = args[i..].to_vec();
	}

	/* was the flag given */
	pub fn
	has(&self, c: char) -> bool
	{
		self.seen.iter().any(|f| f.0 == c)
	}

	/* the last value given for the flag */
	pub fn
	get(&self, c: char) -> Option<&str>
	{
		self.seen.iter().rev().find(|f| f.0 == c).map(|f| f.1.as_str())
	}

	/* every value given for the flag, in order */
	pub fn
	all(&self, c: char) -> Vec<&str>
	{
		self.seen.iter().filter(|f| f.0 == c).map(|f| f.1.as_str()).collect()
	}

	/* the value of the flag as a number, or the default */
	pub fn
	num(&self, c: char, dflt: f64) -> Result<f64, String>
	{
		match self.get(c){
			None => Ok(dflt),
			Some(v) => v.parse::<f64>().map_err(|_| format!("-{} {:?}: not a number", c, v)),
		}
	}

	/* the value of the flag as a whole number, or the default */
	pub fn
	int(&self, c: char, dflt: usize) -> Result<usize, String>
	{
		let v = self.num(c, dflt as f64)?;
		if(v < 0.0 || v.fract() != 0.0){
			return Err(format!("-{} {}: want a whole number", c, v));
		}
		Ok(v as usize)
	}

	/* the flag must have a value */
	pub fn
	need(&self, c: char) -> Result<&str, String>
	{
		match self.get(c){
			Some(v) => Ok(v),
			None => Err(format!("no -{} given", c)),
		}
	}

	pub fn
	usage(&self, syn: &str) -> !
	{
		eprintln!("usage: {} {}", self.prog, syn);
		std::process::exit(2);
	}
}
