# smtdif: how hard an SMT formula is, judged by its shape.
# the pipeline is six filters over text; make takes it from the archives to the tables.
#   make all       the programs, built by cargo
#   make dataset   the dataset table, out of the archives and the competition results
#   make stats     what the dataset holds, in out/stat.txt
#   make fit       every experiment, its tables in out/
#   make calib     the calibration of the predictions by family, out/calib.tsv
#   make worth     what the features are worth, out/imp.tsv and out/ablate.tsv
#   make test      the tests of every stage

CARGO ?= cargo
BIN = target/release
DATA = data
OUT = out
# the programs carry the suffix of the cargo that builds them: .exe on windows, and under wsl
# whenever the cargo at hand is the one of windows
EXE ?= $(if $(filter Windows_NT,$(OS)),.exe,$(suffix $(notdir $(shell command -v $(CARGO) 2>/dev/null))))

# the inputs: the release 2025 of SMT-LIB on Zenodo, and the results of SMT-COMP 2025
RELEASE = https://zenodo.org/api/records/16740866/files
EDITION = 2025
RESULTS = https://raw.githubusercontent.com/SMT-COMP/smt-comp.github.io/master/data/results-sq-$(EDITION).json.gz
LOGICS = QF_BV QF_ABV
SOLVER = cvc5
LIMIT = 1200
SEED = 2026

PROGS = smtfeat smtdata smtstat smtfit smtcal smtimp
BINS = $(PROGS:%=$(BIN)/%$(EXE))
FEATS = $(LOGICS:%=$(DATA)/feat/%.tsv)
RUNS = $(DATA)/results-sq-$(EDITION).json.gz
DATASET = $(DATA)/dataset.tsv
PRED = $(OUT)/pred.tsv
CHOSEN = $(OUT)/time.tsv $(OUT)/outcome.tsv

.PHONY: all features dataset stats fit calib worth test clean

# a stage that fails leaves half a table behind, and half a table must not pass for a whole one
.DELETE_ON_ERROR:

all: $(BINS)

# a program is out of date when the library or its own file changes; cargo builds the six
# together, so whichever is out of date pays for all, and each stage trusts its own program's time
$(BIN)/%$(EXE): $(wildcard src/*.rs) src/bin/%.rs Cargo.toml Cargo.lock
	$(CARGO) build --release

test:
	$(CARGO) test

# ---- inputs

$(DATA)/QF_BV.tar.zst: SIZE = 1732883333
$(DATA)/QF_BV.tar.zst: SUM = 9bb268ced812a004edfd9876aea025bb
$(DATA)/QF_ABV.tar.zst: SIZE = 138845256
$(DATA)/QF_ABV.tar.zst: SUM = ea239b30bbcc7a7f1042b2a98e38351a

# an archive is fetched when it is missing, never again: it is an input, and costs minutes
$(DATA)/%.tar.zst:
	mkdir -p $(DATA)
	sh get.sh $(RELEASE)/$*.tar.zst/content $(SIZE) $@
	echo "$(SUM) $@" | md5sum -c -

$(RUNS):
	mkdir -p $(DATA)
	curl -sS -L -o $@ $(RESULTS)

# ---- the filters

features: $(FEATS)

$(DATA)/feat/%.tsv: $(DATA)/%.tar.zst $(BIN)/smtfeat$(EXE)
	mkdir -p $(DATA)/feat
	$(BIN)/smtfeat$(EXE) $< > $@

dataset: $(DATASET)

$(DATASET): $(FEATS) $(RUNS) $(BIN)/smtdata$(EXE)
	$(BIN)/smtdata$(EXE) -s $(SOLVER) $(LOGICS:%=-l %) $(FEATS:%=-f %) $(RUNS) > $@

stats: $(OUT)/stat.txt

$(OUT)/stat.txt: $(DATASET) $(BIN)/smtstat$(EXE)
	mkdir -p $(OUT)
	$(BIN)/smtstat$(EXE) $< > $@

fit: $(CHOSEN)

# smtfit writes seven tables at once; time.tsv stands for them all
$(OUT)/time.tsv: $(DATASET) $(BIN)/smtfit$(EXE)
	mkdir -p $(OUT)
	$(BIN)/smtfit$(EXE) -r $(SEED) -t $(LIMIT) -o $(OUT) $<

$(OUT)/outcome.tsv $(OUT)/pred.tsv $(OUT)/cv.tsv $(OUT)/conf.tsv $(OUT)/split.tsv: $(OUT)/time.tsv

calib: $(OUT)/calib.tsv

$(OUT)/calib.tsv: $(PRED) $(BIN)/smtcal$(EXE)
	$(BIN)/smtcal$(EXE) $< > $@

worth: $(OUT)/ablate.tsv

$(OUT)/ablate.tsv: $(CHOSEN) $(DATASET) $(BIN)/smtimp$(EXE)
	$(BIN)/smtimp$(EXE) -r $(SEED) -t $(LIMIT) -o $(OUT) $(DATASET)

$(OUT)/imp.tsv: $(OUT)/ablate.tsv

clean:
	$(CARGO) clean
	rm -rf $(OUT)
