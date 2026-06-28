# Confounder coverage summary

- **Total spectra:** 11
- **Indexes:** {'examples_index.sqlite': 11}

## By source

- `examples_reference`: 11

## By acquisition mode

- `unknown`: 11

## By functional-group tag

- `heteroaromatic`: 5
- `phenol`: 2

## Target class counts (true positive / hard negative)

- ✗ `nitro_positive` (true_positive, nitro): **0** / min 10
- ✗ `nitro_hn_n_oxide` (hard_negative, nitro): **0** / min 8
- ✗ `nitro_hn_nitroso` (hard_negative, nitro): **0** / min 5
- ✗ `nitro_hn_heteroaromatic` (hard_negative, nitro): **5** / min 10
- ✗ `nitro_hn_enamine` (hard_negative, nitro): **0** / min 5
- ✗ `amide_positive` (true_positive, amide): **0** / min 10
- ✗ `amide_hn_enamine` (hard_negative, amide): **0** / min 8
- ✗ `amide_hn_pyrrole` (hard_negative, amide): **5** / min 8
- ✗ `amide_hn_imide` (hard_negative, amide): **0** / min 5
- ✗ `amide_hn_conjugated_amide` (hard_negative, amide): **0** / min 5
- ✗ `siloxane_positive` (true_positive, siloxane): **0** / min 8
- ✗ `siloxane_hn_ether_ester` (hard_negative, siloxane): **0** / min 10
- ✗ `siloxane_hn_polymer_co` (hard_negative, siloxane): **0** / min 8
- ✗ `siloxane_hn_atr_polymer` (supporting, siloxane): **0** / min 6

## Coverage gaps (missing spectra)

| class | problem | role | have | min | missing | source |
|-------|---------|------|------|-----|---------|--------|
| amide_positive | amide | true_positive | 0 | 10 | 10 | sdbs_aist |
| nitro_positive | nitro | true_positive | 0 | 10 | 10 | sdbs_aist |
| siloxane_hn_ether_ester | siloxane | hard_negative | 0 | 10 | 10 | sdbs_aist |
| amide_hn_enamine | amide | hard_negative | 0 | 8 | 8 | sdbs_aist |
| nitro_hn_n_oxide | nitro | hard_negative | 0 | 8 | 8 | sdbs_aist |
| siloxane_hn_polymer_co | siloxane | hard_negative | 0 | 8 | 8 | open_polymer_atr |
| siloxane_positive | siloxane | true_positive | 0 | 8 | 8 | open_polymer_atr |
| siloxane_hn_atr_polymer | siloxane | supporting | 0 | 6 | 6 | open_polymer_atr |
| amide_hn_conjugated_amide | amide | hard_negative | 0 | 5 | 5 | sdbs_aist |
| amide_hn_imide | amide | hard_negative | 0 | 5 | 5 | sdbs_aist |
| nitro_hn_enamine | nitro | hard_negative | 0 | 5 | 5 | sdbs_aist |
| nitro_hn_heteroaromatic | nitro | hard_negative | 5 | 10 | 5 | sdbs_aist |
| nitro_hn_nitroso | nitro | hard_negative | 0 | 5 | 5 | sdbs_aist |
| amide_hn_pyrrole | amide | hard_negative | 5 | 8 | 3 | sdbs_aist |
