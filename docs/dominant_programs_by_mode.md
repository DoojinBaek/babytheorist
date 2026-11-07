# BabyTheorist Dominant Program Inspection

- pool_seed: 42
- cache: cache disabled for this inspection run
- direct_max_depth: official
- max_per_depth: 3
- max_parents: 3
- suite-level snapshot: single_easy:1, multi_easy:1, single_hard:7, multi_hard:7
- hard-mode note: add_border first unlocks at level 6; BabyTheoristGym levels 1, 2-3, and 4-5 use smaller hard libraries.
- parent lines show tracked accepted-sample parent id/count histograms when provenance is present.

## single_easy level 1
n_shapes=1 selectors=['by_index'] introduced=None max_depth=10 load_or_build_sec=6.497
effective_primitives=['rot90', 'flip_h', 'flip_v', 'next_color', 'translate_up', 'translate_down', 'translate_left', 'translate_right']
active_segments=['base']

active pool
  depth 1: 8 entries
    01. id=base:d1.01 score=1.0000 s0.A:single.rot90
    02. id=base:d1.02 score=1.0000 s0.A:single.flip_h
    03. id=base:d1.03 score=1.0000 s0.A:single.flip_v
  depth 2: 10 entries
    01. id=base:d2.01 score=0.0250 s0.A:single.flip_h -> s0.A:single.translate_left
        sample_count: 25/1000
        parent: count=25 source=base depth 1 + 1, mode=reuse
          left_id:  base:d1.02
          right_id: base:d1.07
    02. id=base:d2.02 score=0.0250 s0.A:single.rot90 -> s0.A:single.flip_h
        sample_count: 25/1000
        parent: count=25 source=base depth 1 + 1, mode=reuse
          left_id:  base:d1.01
          right_id: base:d1.02
    03. id=base:d2.03 score=0.0250 s0.A:single.translate_right -> s0.A:single.next_color
        sample_count: 25/1000
        parent: count=25 source=base depth 1 + 1, mode=reuse
          left_id:  base:d1.08
          right_id: base:d1.04
  depth 3: 10 entries
    01. id=base:d3.01 score=0.0260 s0.A:single.flip_h -> s0.A:single.rot90 -> s0.A:single.next_color
        sample_count: 26/1000
        parent: count=18 source=base depth 1 + 2, mode=reuse
          left_id:  base:d1.02
          right_id: base:d2.07
        parent: count=8 source=base depth 2 + 1, mode=reuse
          left_id:  base:d2.06
          right_id: base:d1.04
    02. id=base:d3.02 score=0.0220 s0.A:single.flip_h -> s0.A:single.translate_left -> s0.A:single.next_color
        sample_count: 22/1000
        parent: count=13 source=base depth 2 + 1, mode=reuse
          left_id:  base:d2.01
          right_id: base:d1.04
        parent: count=9 source=base depth 1 + 2, mode=reuse
          left_id:  base:d1.02
          right_id: base:d2.08
    03. id=base:d3.03 score=0.0200 s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.flip_h
        sample_count: 20/1000
        parent: count=12 source=base depth 2 + 1, mode=reuse
          left_id:  base:d2.09
          right_id: base:d1.02
        parent: count=8 source=base depth 1 + 2, mode=reuse
          left_id:  base:d1.07
          right_id: base:d2.04
  depth 4: 10 entries
    01. id=base:d4.01 score=0.0250 s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.rot90 -> s0.A:single.next_color
        sample_count: 25/1000
        parent: count=13 source=base depth 2 + 2, mode=reuse
          left_id:  base:d2.04
          right_id: base:d2.07
        parent: count=10 source=base depth 1 + 3, mode=reuse
          left_id:  base:d1.07
          right_id: base:d3.01
        parent: count=2 source=base depth 3 + 1, mode=reuse
          left_id:  base:d3.10
          right_id: base:d1.04
    02. id=base:d4.02 score=0.0240 s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.translate_left -> s0.A:single.next_color
        sample_count: 24/1000
        parent: count=11 source=base depth 1 + 3, mode=reuse
          left_id:  base:d1.07
          right_id: base:d3.02
        parent: count=8 source=base depth 2 + 2, mode=reuse
          left_id:  base:d2.04
          right_id: base:d2.08
        parent: count=5 source=base depth 3 + 1, mode=reuse
          left_id:  base:d3.07
          right_id: base:d1.04
    03. id=base:d4.03 score=0.0190 s0.A:single.rot90 -> s0.A:single.flip_h -> s0.A:single.translate_right -> s0.A:single.next_color
        sample_count: 19/1000
        parent: count=10 source=base depth 1 + 3, mode=reuse
          left_id:  base:d1.01
          right_id: base:d3.05
        parent: count=9 source=base depth 2 + 2, mode=reuse
          left_id:  base:d2.02
          right_id: base:d2.03
  depth 5: 10 entries
    01. id=base:d5.01 score=0.0310 s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.rot90 -> s0.A:single.next_color
        sample_count: 31/1000
        parent: count=11 source=base depth 3 + 2, mode=reuse
          left_id:  base:d3.03
          right_id: base:d2.07
        parent: count=8 source=base depth 4 + 1, mode=reuse
          left_id:  base:d4.05
          right_id: base:d1.04
        parent: count=7 source=base depth 1 + 4, mode=reuse
          left_id:  base:d1.07
          right_id: base:d4.01
        ... 1 more parent paths
    02. id=base:d5.02 score=0.0220 s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.translate_left -> s0.A:single.next_color
        sample_count: 22/1000
        parent: count=11 source=base depth 1 + 4, mode=reuse
          left_id:  base:d1.07
          right_id: base:d4.02
        parent: count=8 source=base depth 3 + 2, mode=reuse
          left_id:  base:d3.03
          right_id: base:d2.08
        parent: count=3 source=base depth 2 + 3, mode=reuse
          left_id:  base:d2.09
          right_id: base:d3.02
    03. id=base:d5.03 score=0.0180 s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.flip_v -> s0.A:single.rot90 -> s0.A:single.next_color
        sample_count: 18/1000
        parent: count=14 source=base depth 2 + 3, mode=reuse
          left_id:  base:d2.04
          right_id: base:d3.06
        parent: count=4 source=base depth 1 + 4, mode=reuse
          left_id:  base:d1.07
          right_id: base:d4.04
  depth 6: 10 entries
    01. id=base:d6.01 score=0.0240 s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.rot90 -> s0.A:single.next_color
        sample_count: 24/1000
        parent: count=13 source=base depth 2 + 4, mode=reuse
          left_id:  base:d2.09
          right_id: base:d4.01
        parent: count=11 source=base depth 1 + 5, mode=reuse
          left_id:  base:d1.07
          right_id: base:d5.01
    02. id=base:d6.02 score=0.0190 s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.rot90 -> s0.A:single.flip_h -> s0.A:single.translate_right -> s0.A:single.next_color
        sample_count: 19/1000
        parent: count=19 source=base depth 1 + 5, mode=reuse
          left_id:  base:d1.08
          right_id: base:d5.09
    03. id=base:d6.03 score=0.0190 s0.A:single.translate_right -> s0.A:single.translate_up -> s0.A:single.rot90 -> s0.A:single.flip_h -> s0.A:single.translate_right -> s0.A:single.next_color
        sample_count: 19/1000
        parent: count=19 source=base depth 1 + 5, mode=reuse
          left_id:  base:d1.08
          right_id: base:d5.10
  depth 7: 10 entries
    01. id=base:d7.01 score=0.0220 s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.rot90 -> s0.A:single.next_color
        sample_count: 22/1000
        parent: count=12 source=base depth 1 + 6, mode=reuse
          left_id:  base:d1.07
          right_id: base:d6.01
        parent: count=10 source=base depth 2 + 5, mode=reuse
          left_id:  base:d2.09
          right_id: base:d5.01
    02. id=base:d7.02 score=0.0220 s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.translate_left -> s0.A:single.next_color
        sample_count: 22/1000
        parent: count=14 source=base depth 1 + 6, mode=reuse
          left_id:  base:d1.07
          right_id: base:d6.05
        parent: count=8 source=base depth 2 + 5, mode=reuse
          left_id:  base:d2.09
          right_id: base:d5.02
    03. id=base:d7.03 score=0.0210 s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.flip_v -> s0.A:single.rot90 -> s0.A:single.next_color
        sample_count: 21/1000
        parent: count=16 source=base depth 1 + 6, mode=reuse
          left_id:  base:d1.07
          right_id: base:d6.04
        parent: count=5 source=base depth 2 + 5, mode=reuse
          left_id:  base:d2.09
          right_id: base:d5.03
  depth 8: 10 entries
    01. id=base:d8.01 score=0.0320 s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.rot90 -> s0.A:single.next_color -> s0.A:single.translate_left -> s0.A:single.translate_down
        sample_count: 32/1000
        parent: count=10 source=base depth 6 + 2, mode=reuse
          left_id:  base:d6.01
          right_id: base:d2.05
        parent: count=9 source=base depth 1 + 7, mode=reuse
          left_id:  base:d1.07
          right_id: base:d7.06
        parent: count=9 source=base depth 2 + 6, mode=reuse
          left_id:  base:d2.09
          right_id: base:d6.09
        ... 1 more parent paths
    02. id=base:d8.02 score=0.0310 s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.rot90 -> s0.A:single.next_color -> s0.A:single.next_color
        sample_count: 31/1000
        parent: count=12 source=base depth 1 + 7, mode=reuse
          left_id:  base:d1.07
          right_id: base:d7.04
        parent: count=11 source=base depth 2 + 6, mode=reuse
          left_id:  base:d2.09
          right_id: base:d6.08
        parent: count=8 source=base depth 7 + 1, mode=reuse
          left_id:  base:d7.01
          right_id: base:d1.04
    03. id=base:d8.03 score=0.0260 s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.rot90 -> s0.A:single.next_color
        sample_count: 26/1000
        parent: count=17 source=base depth 1 + 7, mode=reuse
          left_id:  base:d1.07
          right_id: base:d7.01
        parent: count=9 source=base depth 2 + 6, mode=reuse
          left_id:  base:d2.09
          right_id: base:d6.01
  depth 9: 10 entries
    01. id=base:d9.01 score=0.0360 s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.rot90 -> s0.A:single.flip_h -> s0.A:single.translate_right -> s0.A:single.next_color
        sample_count: 36/1000
        parent: count=36 source=base depth 1 + 8, mode=reuse
          left_id:  base:d1.08
          right_id: base:d8.08
    02. id=base:d9.02 score=0.0350 s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_up -> s0.A:single.rot90 -> s0.A:single.flip_h -> s0.A:single.translate_right -> s0.A:single.next_color
        sample_count: 35/1000
        parent: count=35 source=base depth 1 + 8, mode=reuse
          left_id:  base:d1.08
          right_id: base:d8.05
    03. id=base:d9.03 score=0.0280 s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.rot90 -> s0.A:single.next_color -> s0.A:single.next_color
        sample_count: 28/1000
        parent: count=15 source=base depth 1 + 8, mode=reuse
          left_id:  base:d1.07
          right_id: base:d8.02
        parent: count=11 source=base depth 8 + 1, mode=reuse
          left_id:  base:d8.03
          right_id: base:d1.04
        parent: count=2 source=base depth 2 + 7, mode=reuse
          left_id:  base:d2.09
          right_id: base:d7.04
  depth 10: 10 entries
    01. id=base:d10.01 score=0.0340 s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.rot90 -> s0.A:single.next_color -> s0.A:single.translate_left -> s0.A:single.translate_down
        sample_count: 34/1000
        parent: count=11 source=base depth 1 + 9, mode=reuse
          left_id:  base:d1.07
          right_id: base:d9.04
        parent: count=11 source=base depth 2 + 8, mode=reuse
          left_id:  base:d2.09
          right_id: base:d8.01
        parent: count=7 source=base depth 9 + 1, mode=reuse
          left_id:  base:d9.05
          right_id: base:d1.06
        ... 1 more parent paths
    02. id=base:d10.02 score=0.0320 s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.rot90 -> s0.A:single.flip_h -> s0.A:single.translate_right -> s0.A:single.next_color
        sample_count: 32/1000
        parent: count=32 source=base depth 1 + 9, mode=reuse
          left_id:  base:d1.08
          right_id: base:d9.01
    03. id=base:d10.03 score=0.0290 s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_up -> s0.A:single.rot90 -> s0.A:single.flip_h -> s0.A:single.translate_right -> s0.A:single.next_color
        sample_count: 29/1000
        parent: count=29 source=base depth 1 + 9, mode=reuse
          left_id:  base:d1.08
          right_id: base:d9.02

## multi_easy level 1
n_shapes=2 selectors=['by_index'] introduced=None max_depth=10 load_or_build_sec=13.450
effective_primitives=['rot90', 'flip_h', 'flip_v', 'next_color', 'translate_up', 'translate_down', 'translate_left', 'translate_right', 'swap_colors', 'align_row', 'align_col']
active_segments=['base']

active pool
  depth 1: 11 entries
    01. id=base:d1.01 score=1.0000 s0.A:single.rot90
    02. id=base:d1.02 score=1.0000 s0.A:single.flip_h
    03. id=base:d1.03 score=1.0000 s0.A:single.flip_v
  depth 2: 10 entries
    01. id=base:d2.01 score=0.0190 s0.A:single.flip_h -> s0.A:single.align_col(s0.B:single)
        sample_count: 19/1000
        parent: count=19 source=base depth 1 + 1, mode=reuse
          left_id:  base:d1.02
          right_id: base:d1.11
    02. id=base:d2.02 score=0.0180 s0.A:single.translate_down -> s0.A:single.align_col(s0.B:single)
        sample_count: 18/1000
        parent: count=18 source=base depth 1 + 1, mode=reuse
          left_id:  base:d1.06
          right_id: base:d1.11
    03. id=base:d2.03 score=0.0140 s0.A:single.flip_h -> s0.A:single.align_row(s0.B:single)
        sample_count: 14/1000
        parent: count=14 source=base depth 1 + 1, mode=reuse
          left_id:  base:d1.02
          right_id: base:d1.10
  depth 3: 10 entries
    01. id=base:d3.01 score=0.0190 s0.A:single.swap_colors(s0.B:single) -> s0.A:single.align_row(s0.B:single) -> s0.A:single.flip_v
        sample_count: 19/1000
        parent: count=10 source=base depth 1 + 2, mode=reuse
          left_id:  base:d1.09
          right_id: base:d2.07
        parent: count=9 source=base depth 2 + 1, mode=reuse
          left_id:  base:d2.04
          right_id: base:d1.03
    02. id=base:d3.02 score=0.0160 s0.A:single.flip_h -> s0.A:single.translate_down -> s0.A:single.align_col(s0.B:single)
        sample_count: 16/1000
        parent: count=16 source=base depth 1 + 2, mode=reuse
          left_id:  base:d1.02
          right_id: base:d2.02
    03. id=base:d3.03 score=0.0140 s0.A:single.align_row(s0.B:single) -> s0.A:single.swap_colors(s0.B:single) -> s0.A:single.translate_up
        sample_count: 14/1000
        parent: count=9 source=base depth 2 + 1, mode=reuse
          left_id:  base:d2.08
          right_id: base:d1.05
        parent: count=5 source=base depth 1 + 2, mode=reuse
          left_id:  base:d1.10
          right_id: base:d2.10
  depth 4: 10 entries
    01. id=base:d4.01 score=0.0280 s0.A:single.flip_h -> s0.A:single.align_row(s0.B:single) -> s0.A:single.swap_colors(s0.B:single) -> s0.A:single.translate_up
        sample_count: 28/1000
        parent: count=12 source=base depth 1 + 3, mode=reuse
          left_id:  base:d1.02
          right_id: base:d3.03
        parent: count=9 source=base depth 3 + 1, mode=reuse
          left_id:  base:d3.07
          right_id: base:d1.05
        parent: count=7 source=base depth 2 + 2, mode=reuse
          left_id:  base:d2.03
          right_id: base:d2.10
    02. id=base:d4.02 score=0.0240 s0.A:single.flip_h -> s0.A:single.translate_down -> s0.A:single.align_col(s0.B:single) -> s0.A:single.translate_right
        sample_count: 24/1000
        parent: count=12 source=base depth 1 + 3, mode=reuse
          left_id:  base:d1.02
          right_id: base:d3.05
        parent: count=12 source=base depth 3 + 1, mode=reuse
          left_id:  base:d3.02
          right_id: base:d1.08
    03. id=base:d4.03 score=0.0190 s0.A:single.align_row(s0.B:single) -> s0.A:single.swap_colors(s0.B:single) -> s0.A:single.translate_down -> s0.A:single.align_col(s0.B:single)
        sample_count: 19/1000
        parent: count=14 source=base depth 2 + 2, mode=reuse
          left_id:  base:d2.08
          right_id: base:d2.02
        parent: count=5 source=base depth 1 + 3, mode=reuse
          left_id:  base:d1.10
          right_id: base:d3.08
  depth 5: 10 entries
    01. id=base:d5.01 score=0.0330 s0.A:single.flip_h -> s0.A:single.align_row(s0.B:single) -> s0.A:single.swap_colors(s0.B:single) -> s0.A:single.translate_down -> s0.A:single.align_col(s0.B:single)
        sample_count: 33/1000
        parent: count=16 source=base depth 1 + 4, mode=reuse
          left_id:  base:d1.02
          right_id: base:d4.03
        parent: count=10 source=base depth 3 + 2, mode=reuse
          left_id:  base:d3.07
          right_id: base:d2.02
        parent: count=7 source=base depth 2 + 3, mode=reuse
          left_id:  base:d2.03
          right_id: base:d3.08
    02. id=base:d5.02 score=0.0270 s0.A:single.flip_h -> s0.A:single.align_col(s0.B:single) -> s0.A:single.swap_colors(s0.B:single) -> s0.A:single.align_row(s0.B:single) -> s0.A:single.flip_v
        sample_count: 27/1000
        parent: count=11 source=base depth 2 + 3, mode=reuse
          left_id:  base:d2.01
          right_id: base:d3.01
        parent: count=10 source=base depth 3 + 2, mode=reuse
          left_id:  base:d3.09
          right_id: base:d2.07
        parent: count=6 source=base depth 4 + 1, mode=reuse
          left_id:  base:d4.04
          right_id: base:d1.03
    03. id=base:d5.03 score=0.0210 s0.A:single.align_row(s0.B:single) -> s0.A:single.flip_h -> s0.A:single.translate_down -> s0.A:single.align_col(s0.B:single) -> s0.A:single.translate_right
        sample_count: 21/1000
        parent: count=11 source=base depth 1 + 4, mode=reuse
          left_id:  base:d1.10
          right_id: base:d4.02
        parent: count=10 source=base depth 2 + 3, mode=reuse
          left_id:  base:d2.06
          right_id: base:d3.05
  depth 6: 10 entries
    01. id=base:d6.01 score=0.0360 s0.A:single.flip_h -> s0.A:single.align_row(s0.B:single) -> s0.A:single.swap_colors(s0.B:single) -> s0.A:single.translate_down -> s0.A:single.align_col(s0.B:single) -> s0.A:single.translate_right
        sample_count: 36/1000
        parent: count=14 source=base depth 5 + 1, mode=reuse
          left_id:  base:d5.01
          right_id: base:d1.08
        parent: count=11 source=base depth 1 + 5, mode=reuse
          left_id:  base:d1.02
          right_id: base:d5.04
        parent: count=11 source=base depth 3 + 3, mode=reuse
          left_id:  base:d3.07
          right_id: base:d3.05
    02. id=base:d6.02 score=0.0230 s0.A:single.swap_colors(s0.B:single) -> s0.A:single.align_row(s0.B:single) -> s0.A:single.flip_h -> s0.A:single.translate_down -> s0.A:single.align_col(s0.B:single) -> s0.A:single.translate_right
        sample_count: 23/1000
        parent: count=15 source=base depth 2 + 4, mode=reuse
          left_id:  base:d2.04
          right_id: base:d4.02
        parent: count=8 source=base depth 1 + 5, mode=reuse
          left_id:  base:d1.09
          right_id: base:d5.03
    03. id=base:d6.03 score=0.0220 s0.A:single.flip_h -> s0.A:single.align_col(s0.B:single) -> s0.A:single.translate_left -> s0.A:single.swap_colors(s0.B:single) -> s0.A:single.align_row(s0.B:single) -> s0.A:single.flip_v
        sample_count: 22/1000
        parent: count=13 source=base depth 2 + 4, mode=reuse
          left_id:  base:d2.01
          right_id: base:d4.10
        parent: count=9 source=base depth 3 + 3, mode=reuse
          left_id:  base:d3.10
          right_id: base:d3.01
  depth 7: 10 entries
    01. id=base:d7.01 score=0.0260 s0.A:single.align_row(s0.B:single) -> s0.A:single.flip_h -> s0.A:single.translate_down -> s0.A:single.align_col(s0.B:single) -> s0.A:single.translate_right -> s0.A:single.swap_colors(s0.B:single) -> s0.A:single.translate_right
        sample_count: 26/1000
        parent: count=15 source=base depth 5 + 2, mode=reuse
          left_id:  base:d5.03
          right_id: base:d2.05
        parent: count=11 source=base depth 1 + 6, mode=reuse
          left_id:  base:d1.10
          right_id: base:d6.09
    02. id=base:d7.02 score=0.0260 s0.A:single.swap_colors(s0.B:single) -> s0.A:single.align_row(s0.B:single) -> s0.A:single.flip_h -> s0.A:single.translate_down -> s0.A:single.align_col(s0.B:single) -> s0.A:single.translate_right -> s0.A:single.flip_v
        sample_count: 26/1000
        parent: count=11 source=base depth 6 + 1, mode=reuse
          left_id:  base:d6.02
          right_id: base:d1.03
        parent: count=9 source=base depth 1 + 6, mode=reuse
          left_id:  base:d1.09
          right_id: base:d6.10
        parent: count=6 source=base depth 2 + 5, mode=reuse
          left_id:  base:d2.04
          right_id: base:d5.09
    03. id=base:d7.03 score=0.0260 s0.A:single.swap_colors(s0.B:single) -> s0.A:single.align_row(s0.B:single) -> s0.A:single.flip_v -> s0.A:single.flip_h -> s0.A:single.translate_down -> s0.A:single.align_col(s0.B:single) -> s0.A:single.translate_right
        sample_count: 26/1000
        parent: count=22 source=base depth 3 + 4, mode=reuse
          left_id:  base:d3.01
          right_id: base:d4.02
        parent: count=4 source=base depth 6 + 1, mode=reuse
          left_id:  base:d6.05
          right_id: base:d1.08
  depth 8: 10 entries
    01. id=base:d8.01 score=0.0410 s0.A:single.next_color -> s0.A:single.swap_colors(s0.B:single) -> s0.A:single.align_row(s0.B:single) -> s0.A:single.flip_v -> s0.A:single.flip_h -> s0.A:single.translate_down -> s0.A:single.align_col(s0.B:single) -> s0.A:single.translate_right
        sample_count: 41/1000
        parent: count=18 source=base depth 1 + 7, mode=reuse
          left_id:  base:d1.04
          right_id: base:d7.03
        parent: count=14 source=base depth 7 + 1, mode=reuse
          left_id:  base:d7.05
          right_id: base:d1.08
        parent: count=9 source=base depth 4 + 4, mode=reuse
          left_id:  base:d4.09
          right_id: base:d4.02
    02. id=base:d8.02 score=0.0300 s0.A:single.swap_colors(s0.B:single) -> s0.A:single.align_row(s0.B:single) -> s0.A:single.flip_h -> s0.A:single.translate_down -> s0.A:single.align_col(s0.B:single) -> s0.A:single.translate_right -> s0.A:single.flip_v -> s0.A:single.rot90
        sample_count: 30/1000
        parent: count=30 source=base depth 7 + 1, mode=reuse
          left_id:  base:d7.02
          right_id: base:d1.01
    03. id=base:d8.03 score=0.0250 s0.A:single.align_row(s0.B:single) -> s0.A:single.flip_h -> s0.A:single.translate_down -> s0.A:single.align_col(s0.B:single) -> s0.A:single.translate_right -> s0.A:single.swap_colors(s0.B:single) -> s0.A:single.translate_right -> s0.A:single.translate_down
        sample_count: 25/1000
        parent: count=25 source=base depth 7 + 1, mode=reuse
          left_id:  base:d7.01
          right_id: base:d1.06
  depth 9: 10 entries
    01. id=base:d9.01 score=0.0350 s0.A:single.next_color -> s0.A:single.swap_colors(s0.B:single) -> s0.A:single.align_row(s0.B:single) -> s0.A:single.flip_v -> s0.A:single.flip_h -> s0.A:single.translate_down -> s0.A:single.align_col(s0.B:single) -> s0.A:single.translate_right -> s0.A:single.translate_down
        sample_count: 35/1000
        parent: count=24 source=base depth 8 + 1, mode=reuse
          left_id:  base:d8.01
          right_id: base:d1.06
        parent: count=11 source=base depth 1 + 8, mode=reuse
          left_id:  base:d1.04
          right_id: base:d8.09
    02. id=base:d9.02 score=0.0300 s0.A:single.next_color -> s0.A:single.swap_colors(s0.B:single) -> s0.A:single.align_row(s0.B:single) -> s0.A:single.flip_v -> s0.A:single.flip_h -> s0.A:single.translate_down -> s0.A:single.align_col(s0.B:single) -> s0.A:single.translate_right -> s0.A:single.rot90
        sample_count: 30/1000
        parent: count=30 source=base depth 8 + 1, mode=reuse
          left_id:  base:d8.01
          right_id: base:d1.01
    03. id=base:d9.03 score=0.0300 s0.A:single.rot90 -> s0.A:single.align_row(s0.B:single) -> s0.A:single.flip_h -> s0.A:single.translate_down -> s0.A:single.align_col(s0.B:single) -> s0.A:single.translate_right -> s0.A:single.swap_colors(s0.B:single) -> s0.A:single.translate_right -> s0.A:single.translate_down
        sample_count: 30/1000
        parent: count=17 source=base depth 1 + 8, mode=reuse
          left_id:  base:d1.01
          right_id: base:d8.03
        parent: count=13 source=base depth 8 + 1, mode=reuse
          left_id:  base:d8.08
          right_id: base:d1.06
  depth 10: 10 entries
    01. id=base:d10.01 score=0.0370 s0.A:single.next_color -> s0.A:single.next_color -> s0.A:single.swap_colors(s0.B:single) -> s0.A:single.align_row(s0.B:single) -> s0.A:single.flip_v -> s0.A:single.flip_h -> s0.A:single.translate_down -> s0.A:single.align_col(s0.B:single) -> s0.A:single.translate_right -> s0.A:single.translate_down
        sample_count: 37/1000
        parent: count=19 source=base depth 9 + 1, mode=reuse
          left_id:  base:d9.09
          right_id: base:d1.06
        parent: count=18 source=base depth 1 + 9, mode=reuse
          left_id:  base:d1.04
          right_id: base:d9.01
    02. id=base:d10.02 score=0.0330 s0.A:single.next_color -> s0.A:single.next_color -> s0.A:single.swap_colors(s0.B:single) -> s0.A:single.align_row(s0.B:single) -> s0.A:single.flip_v -> s0.A:single.flip_h -> s0.A:single.translate_down -> s0.A:single.align_col(s0.B:single) -> s0.A:single.translate_right -> s0.A:single.translate_right
        sample_count: 33/1000
        parent: count=19 source=base depth 9 + 1, mode=reuse
          left_id:  base:d9.09
          right_id: base:d1.08
        parent: count=14 source=base depth 1 + 9, mode=reuse
          left_id:  base:d1.04
          right_id: base:d9.05
    03. id=base:d10.03 score=0.0330 s0.A:single.rot90 -> s0.A:single.align_row(s0.B:single) -> s0.A:single.flip_h -> s0.A:single.translate_down -> s0.A:single.align_col(s0.B:single) -> s0.A:single.translate_right -> s0.A:single.swap_colors(s0.B:single) -> s0.A:single.translate_right -> s0.A:single.translate_down -> s0.A:single.translate_down
        sample_count: 33/1000
        parent: count=19 source=base depth 9 + 1, mode=reuse
          left_id:  base:d9.03
          right_id: base:d1.06
        parent: count=14 source=base depth 1 + 9, mode=reuse
          left_id:  base:d1.01
          right_id: base:d9.06

## single_hard level 7
n_shapes=1 selectors=['by_index'] introduced=add_border max_depth=10 load_or_build_sec=28.172
effective_primitives=['rot90', 'flip_h', 'flip_v', 'next_color', 'translate_up', 'translate_down', 'translate_left', 'translate_right', 'hollow', 'fill_interior', 'add_border']
active_segments=['base', 'append:hollow', 'append:fill_interior', 'append:add_border']

active pool
  depth 1: 11 entries
    01. id=base:d1.01 score=1.0000 s0.A:single.rot90
    02. id=base:d1.02 score=1.0000 s0.A:single.flip_h
    03. id=base:d1.03 score=1.0000 s0.A:single.flip_v
  depth 2: 25 entries
    01. id=append:fill_interior:d2.01 score=0.0830 s0.A:single.next_color -> s0.A:single.fill_interior
        sample_count: 83/1000
        parent: count=83 source=append_right depth 1 + 1, mode=reuse
          left_id:  base:d1.04
          right_id: append:fill_interior:d1.01
    02. id=append:hollow:d2.01 score=0.0730 s0.A:single.hollow -> s0.A:single.flip_v
        sample_count: 73/1000
        parent: count=73 source=append_left depth 1 + 1, mode=reuse
          left_id:  append:hollow:d1.01
          right_id: base:d1.03
    03. id=append:hollow:d2.02 score=0.0730 s0.A:single.hollow -> s0.A:single.rot90
        sample_count: 73/1000
        parent: count=73 source=append_left depth 1 + 1, mode=reuse
          left_id:  append:hollow:d1.01
          right_id: base:d1.01
  depth 3: 25 entries
    01. id=append:hollow:d3.01 score=0.0400 s0.A:single.hollow -> s0.A:single.rot90 -> s0.A:single.flip_h
        sample_count: 40/1000
        parent: count=29 source=append_left depth 1 + 2, mode=reuse
          left_id:  append:hollow:d1.01
          right_id: base:d2.02
        parent: count=11 source=append_left depth 2 + 1, mode=reuse
          left_id:  append:hollow:d2.02
          right_id: base:d1.02
    02. id=append:hollow:d3.02 score=0.0370 s0.A:single.rot90 -> s0.A:single.flip_h -> s0.A:single.hollow
        sample_count: 37/1000
        parent: count=37 source=append_right depth 2 + 1, mode=reuse
          left_id:  base:d2.02
          right_id: append:hollow:d1.01
    03. id=append:hollow:d3.03 score=0.0360 s0.A:single.rot90 -> s0.A:single.next_color -> s0.A:single.hollow
        sample_count: 36/1000
        parent: count=29 source=append_right depth 2 + 1, mode=reuse
          left_id:  base:d2.07
          right_id: append:hollow:d1.01
        parent: count=7 source=append_right depth 1 + 2, mode=reuse
          left_id:  base:d1.01
          right_id: append:hollow:d2.05
  depth 4: 25 entries
    01. id=append:fill_interior:d4.01 score=0.0350 s0.A:single.flip_h -> s0.A:single.translate_left -> s0.A:single.next_color -> s0.A:single.fill_interior
        sample_count: 35/1000
        parent: count=18 source=append_right depth 3 + 1, mode=reuse
          left_id:  base:d3.02
          right_id: append:fill_interior:d1.01
        parent: count=11 source=append_right depth 1 + 3, mode=reuse
          left_id:  base:d1.02
          right_id: append:fill_interior:d3.02
        parent: count=6 source=append_right depth 2 + 2, mode=reuse
          left_id:  base:d2.01
          right_id: append:fill_interior:d2.01
    02. id=append:hollow:d4.01 score=0.0310 s0.A:single.flip_h -> s0.A:single.translate_left -> s0.A:single.next_color -> s0.A:single.hollow
        sample_count: 31/1000
        parent: count=27 source=append_right depth 3 + 1, mode=reuse
          left_id:  base:d3.02
          right_id: append:hollow:d1.01
        parent: count=4 source=append_right depth 2 + 2, mode=reuse
          left_id:  base:d2.01
          right_id: append:hollow:d2.05
    03. id=append:hollow:d4.02 score=0.0310 s0.A:single.hollow -> s0.A:single.flip_h -> s0.A:single.rot90 -> s0.A:single.next_color
        sample_count: 31/1000
        parent: count=31 source=append_left depth 1 + 3, mode=reuse
          left_id:  append:hollow:d1.01
          right_id: base:d3.01
  depth 5: 25 entries
    01. id=append:fill_interior:d5.01 score=0.0350 s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.translate_left -> s0.A:single.next_color -> s0.A:single.fill_interior
        sample_count: 35/1000
        parent: count=16 source=append_right depth 4 + 1, mode=reuse
          left_id:  base:d4.02
          right_id: append:fill_interior:d1.01
        parent: count=12 source=append_right depth 1 + 4, mode=reuse
          left_id:  base:d1.07
          right_id: append:fill_interior:d4.01
        parent: count=6 source=append_right depth 3 + 2, mode=reuse
          left_id:  base:d3.07
          right_id: append:fill_interior:d2.01
        ... 1 more parent paths
    02. id=append:hollow:d5.01 score=0.0330 s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.translate_left -> s0.A:single.next_color -> s0.A:single.hollow
        sample_count: 33/1000
        parent: count=22 source=append_right depth 4 + 1, mode=reuse
          left_id:  base:d4.02
          right_id: append:hollow:d1.01
        parent: count=7 source=append_right depth 1 + 4, mode=reuse
          left_id:  base:d1.07
          right_id: append:hollow:d4.01
        parent: count=4 source=append_right depth 3 + 2, mode=reuse
          left_id:  base:d3.07
          right_id: append:hollow:d2.05
    03. id=base:d5.01 score=0.0310 s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.rot90 -> s0.A:single.next_color
        sample_count: 31/1000
        parent: count=11 source=base depth 3 + 2, mode=reuse
          left_id:  base:d3.03
          right_id: base:d2.07
        parent: count=8 source=base depth 4 + 1, mode=reuse
          left_id:  base:d4.05
          right_id: base:d1.04
        parent: count=7 source=base depth 1 + 4, mode=reuse
          left_id:  base:d1.07
          right_id: base:d4.01
        ... 1 more parent paths
  depth 6: 25 entries
    01. id=append:hollow:d6.01 score=0.0530 s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.rot90 -> s0.A:single.next_color -> s0.A:single.hollow
        sample_count: 53/1000
        parent: count=39 source=append_right depth 5 + 1, mode=reuse
          left_id:  base:d5.01
          right_id: append:hollow:d1.01
        parent: count=6 source=append_right depth 4 + 2, mode=reuse
          left_id:  base:d4.05
          right_id: append:hollow:d2.05
        parent: count=4 source=append_right depth 3 + 3, mode=reuse
          left_id:  base:d3.03
          right_id: append:hollow:d3.03
        ... 2 more parent paths
    02. id=append:fill_interior:d6.01 score=0.0350 s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.rot90 -> s0.A:single.next_color -> s0.A:single.fill_interior
        sample_count: 35/1000
        parent: count=25 source=append_right depth 5 + 1, mode=reuse
          left_id:  base:d5.01
          right_id: append:fill_interior:d1.01
        parent: count=7 source=append_right depth 1 + 5, mode=reuse
          left_id:  base:d1.07
          right_id: append:fill_interior:d5.02
        parent: count=2 source=append_right depth 4 + 2, mode=reuse
          left_id:  base:d4.05
          right_id: append:fill_interior:d2.01
        ... 1 more parent paths
    03. id=append:fill_interior:d6.02 score=0.0350 s0.A:single.translate_right -> s0.A:single.rot90 -> s0.A:single.flip_h -> s0.A:single.translate_right -> s0.A:single.next_color -> s0.A:single.fill_interior
        sample_count: 35/1000
        parent: count=20 source=append_right depth 1 + 5, mode=reuse
          left_id:  base:d1.08
          right_id: append:fill_interior:d5.03
        parent: count=15 source=append_right depth 5 + 1, mode=reuse
          left_id:  base:d5.09
          right_id: append:fill_interior:d1.01
  depth 7: 25 entries
    01. id=append:fill_interior:d7.01 score=0.0360 s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.rot90 -> s0.A:single.flip_h -> s0.A:single.translate_right -> s0.A:single.next_color -> s0.A:single.fill_interior
        sample_count: 36/1000
        parent: count=21 source=append_right depth 6 + 1, mode=reuse
          left_id:  base:d6.02
          right_id: append:fill_interior:d1.01
        parent: count=15 source=append_right depth 1 + 6, mode=reuse
          left_id:  base:d1.08
          right_id: append:fill_interior:d6.02
    02. id=append:hollow:d7.01 score=0.0350 s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.rot90 -> s0.A:single.next_color -> s0.A:single.hollow
        sample_count: 35/1000
        parent: count=22 source=append_right depth 6 + 1, mode=reuse
          left_id:  base:d6.01
          right_id: append:hollow:d1.01
        parent: count=9 source=append_right depth 1 + 6, mode=reuse
          left_id:  base:d1.07
          right_id: append:hollow:d6.01
        parent: count=4 source=append_right depth 2 + 5, mode=reuse
          left_id:  base:d2.09
          right_id: append:hollow:d5.04
    03. id=append:fill_interior:d7.02 score=0.0340 s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.translate_left -> s0.A:single.next_color -> s0.A:single.fill_interior
        sample_count: 34/1000
        parent: count=19 source=append_right depth 6 + 1, mode=reuse
          left_id:  base:d6.05
          right_id: append:fill_interior:d1.01
        parent: count=11 source=append_right depth 1 + 6, mode=reuse
          left_id:  base:d1.07
          right_id: append:fill_interior:d6.04
        parent: count=4 source=append_right depth 2 + 5, mode=reuse
          left_id:  base:d2.09
          right_id: append:fill_interior:d5.01
  depth 8: 25 entries
    01. id=append:hollow:d8.01 score=0.0400 s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.rot90 -> s0.A:single.next_color -> s0.A:single.next_color -> s0.A:single.hollow
        sample_count: 40/1000
        parent: count=30 source=append_right depth 7 + 1, mode=reuse
          left_id:  base:d7.04
          right_id: append:hollow:d1.01
        parent: count=10 source=append_right depth 6 + 2, mode=reuse
          left_id:  base:d6.01
          right_id: append:hollow:d2.05
    02. id=append:fill_interior:d8.01 score=0.0370 s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.rot90 -> s0.A:single.flip_h -> s0.A:single.translate_right -> s0.A:single.next_color -> s0.A:single.fill_interior
        sample_count: 37/1000
        parent: count=19 source=append_right depth 7 + 1, mode=reuse
          left_id:  base:d7.08
          right_id: append:fill_interior:d1.01
        parent: count=18 source=append_right depth 1 + 7, mode=reuse
          left_id:  base:d1.08
          right_id: append:fill_interior:d7.01
    03. id=append:fill_interior:d8.02 score=0.0350 s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.rot90 -> s0.A:single.next_color -> s0.A:single.next_color -> s0.A:single.fill_interior
        sample_count: 35/1000
        parent: count=22 source=append_right depth 7 + 1, mode=reuse
          left_id:  base:d7.04
          right_id: append:fill_interior:d1.01
        parent: count=8 source=append_right depth 1 + 7, mode=reuse
          left_id:  base:d1.07
          right_id: append:fill_interior:d7.04
        parent: count=5 source=append_right depth 6 + 2, mode=reuse
          left_id:  base:d6.01
          right_id: append:fill_interior:d2.01
  depth 9: 25 entries
    01. id=append:hollow:d9.01 score=0.0380 s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.rot90 -> s0.A:single.next_color -> s0.A:single.translate_left -> s0.A:single.translate_down -> s0.A:single.hollow
        sample_count: 38/1000
        parent: count=38 source=append_right depth 8 + 1, mode=reuse
          left_id:  base:d8.01
          right_id: append:hollow:d1.01
    02. id=append:fill_interior:d9.01 score=0.0370 s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.translate_left -> s0.A:single.flip_h -> s0.A:single.rot90 -> s0.A:single.next_color -> s0.A:single.fill_interior
        sample_count: 37/1000
        parent: count=28 source=append_right depth 8 + 1, mode=reuse
          left_id:  base:d8.03
          right_id: append:fill_interior:d1.01
        parent: count=5 source=append_right depth 2 + 7, mode=reuse
          left_id:  base:d2.09
          right_id: append:fill_interior:d7.03
        parent: count=4 source=append_right depth 1 + 8, mode=reuse
          left_id:  base:d1.07
          right_id: append:fill_interior:d8.05
    03. id=base:d9.01 score=0.0360 s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.rot90 -> s0.A:single.flip_h -> s0.A:single.translate_right -> s0.A:single.next_color
        sample_count: 36/1000
        parent: count=36 source=base depth 1 + 8, mode=reuse
          left_id:  base:d1.08
          right_id: base:d8.08
  depth 10: 25 entries
    01. id=append:fill_interior:d10.01 score=0.0530 s0.A:single.fill_interior -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_up -> s0.A:single.rot90 -> s0.A:single.flip_h -> s0.A:single.translate_right -> s0.A:single.next_color
        sample_count: 53/1000
        parent: count=29 source=append_left depth 1 + 9, mode=reuse
          left_id:  append:fill_interior:d1.01
          right_id: base:d9.02
        parent: count=24 source=append_left depth 2 + 8, mode=reuse
          left_id:  append:fill_interior:d2.05
          right_id: base:d8.05
    02. id=append:fill_interior:d10.02 score=0.0490 s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.rot90 -> s0.A:single.flip_h -> s0.A:single.translate_right -> s0.A:single.next_color -> s0.A:single.fill_interior
        sample_count: 49/1000
        parent: count=34 source=append_right depth 9 + 1, mode=reuse
          left_id:  base:d9.01
          right_id: append:fill_interior:d1.01
        parent: count=15 source=append_right depth 1 + 9, mode=reuse
          left_id:  base:d1.08
          right_id: append:fill_interior:d9.04
    03. id=append:fill_interior:d10.03 score=0.0400 s0.A:single.fill_interior -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.translate_right -> s0.A:single.rot90 -> s0.A:single.flip_h -> s0.A:single.translate_right -> s0.A:single.next_color
        sample_count: 40/1000
        parent: count=22 source=append_left depth 1 + 9, mode=reuse
          left_id:  append:fill_interior:d1.01
          right_id: base:d9.01
        parent: count=18 source=append_left depth 2 + 8, mode=reuse
          left_id:  append:fill_interior:d2.05
          right_id: base:d8.08

## multi_hard level 7
n_shapes=3 selectors=['ALL', 'by_index', 'by_color', 'is_hollow', 'topmost', 'bottommost', 'leftmost', 'rightmost'] introduced=add_border max_depth=10 load_or_build_sec=54.341
effective_primitives=['rot90', 'flip_h', 'flip_v', 'next_color', 'translate_up', 'translate_down', 'translate_left', 'translate_right', 'swap_colors', 'align_row', 'align_col', 'hollow', 'fill_interior', 'add_border']
active_segments=['base', 'append:hollow', 'append:fill_interior', 'append:add_border']

active pool
  depth 1: 25 entries
    01. id=base:d1.01 score=1.0000 s0.A:single.rot90
    02. id=base:d1.02 score=1.0000 s0.G:group.rot90
    03. id=base:d1.03 score=1.0000 s0.A:single.flip_h
  depth 2: 25 entries
    01. id=append:fill_interior:d2.01 score=0.0170 s0.A:single.fill_interior -> s0.A:single.flip_h
        sample_count: 17/1000
        parent: count=17 source=append_left depth 1 + 1, mode=reuse
          left_id:  append:fill_interior:d1.01
          right_id: base:d1.03
    02. id=append:hollow:d2.01 score=0.0150 s0.A:single.flip_v -> s0.G:group.hollow
        sample_count: 15/1000
        parent: count=15 source=append_right depth 1 + 1, mode=reuse
          left_id:  base:d1.05
          right_id: append:hollow:d1.02
    03. id=append:fill_interior:d2.02 score=0.0130 s0.A:single.swap_colors(s0.B:single) -> s1.A:single.fill_interior
        sample_count: 13/1000
        parent: count=13 source=append_right depth 1 + 1, mode=fresh
          left_id:  base:d1.17
          right_id: append:fill_interior:d1.01
  depth 3: 25 entries
    01. id=append:hollow:d3.01 score=0.0140 s0.A:single.hollow -> s0.A:single.swap_colors(s0.B:single) -> s0.G:group.translate_down
        sample_count: 14/1000
        parent: count=14 source=append_left depth 1 + 2, mode=reuse
          left_id:  append:hollow:d1.01
          right_id: base:d2.01
    02. id=append:hollow:d3.02 score=0.0130 s0.A:single.hollow -> s0.A:single.flip_v -> s1.A:single.align_row(s1.B:single)
        sample_count: 13/1000
        parent: count=13 source=append_left depth 1 + 2, mode=reuse
          left_id:  append:hollow:d1.01
          right_id: base:d2.05
    03. id=append:hollow:d3.03 score=0.0130 s0.A:single.swap_colors(s0.B:single) -> s0.G:group.translate_down -> s1.G:group.hollow
        sample_count: 13/1000
        parent: count=13 source=append_right depth 2 + 1, mode=fresh
          left_id:  base:d2.01
          right_id: append:hollow:d1.02
  depth 4: 25 entries
    01. id=append:hollow:d4.01 score=0.0160 s0.A:single.hollow -> s0.A:single.flip_v -> s1.A:single.align_row(s1.B:single) -> s0.A:single.translate_left
        sample_count: 16/1000
        parent: count=13 source=append_left depth 1 + 3, mode=reuse
          left_id:  append:hollow:d1.01
          right_id: base:d3.03
        parent: count=3 source=append_left depth 3 + 1, mode=reuse
          left_id:  append:hollow:d3.02
          right_id: base:d1.13
    02. id=append:fill_interior:d4.01 score=0.0160 s0.A:single.fill_interior -> s0.A:single.align_row(s0.B:single) -> s0.A:single.rot90 -> s1.A:single.translate_up
        sample_count: 16/1000
        parent: count=13 source=append_left depth 1 + 3, mode=reuse
          left_id:  append:fill_interior:d1.01
          right_id: base:d3.01
        parent: count=3 source=append_left depth 2 + 2, mode=reuse
          left_id:  append:fill_interior:d2.04
          right_id: base:d2.06
    03. id=append:fill_interior:d4.02 score=0.0160 s0.A:single.fill_interior -> s0.A:single.translate_down -> s0.A:single.flip_v -> s1.A:single.align_row(s1.B:single)
        sample_count: 16/1000
        parent: count=16 source=append_left depth 1 + 3, mode=reuse
          left_id:  append:fill_interior:d1.01
          right_id: base:d3.05
  depth 5: 25 entries
    01. id=append:fill_interior:d5.01 score=0.0210 s0.A:single.fill_interior -> s0.A:single.align_row(s0.B:single) -> s0.A:single.rot90 -> s1.A:single.translate_up -> s0.A:single.translate_left
        sample_count: 21/1000
        parent: count=11 source=append_left depth 4 + 1, mode=reuse
          left_id:  append:fill_interior:d4.01
          right_id: base:d1.13
        parent: count=10 source=append_left depth 1 + 4, mode=reuse
          left_id:  append:fill_interior:d1.01
          right_id: base:d4.07
    02. id=append:fill_interior:d5.02 score=0.0210 s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.flip_v -> s1.A:single.align_row(s1.B:single) -> s0.A:single.fill_interior
        sample_count: 21/1000
        parent: count=15 source=append_right depth 4 + 1, mode=reuse
          left_id:  base:d4.02
          right_id: append:fill_interior:d1.01
        parent: count=6 source=append_right depth 2 + 3, mode=reuse
          left_id:  base:d2.02
          right_id: append:fill_interior:d3.02
    03. id=base:d5.01 score=0.0170 s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.flip_v -> s1.A:single.align_row(s1.B:single)
        sample_count: 17/1000
        parent: count=7 source=base depth 1 + 4, mode=reuse
          left_id:  base:d1.11
          right_id: base:d4.02
        parent: count=6 source=base depth 3 + 2, mode=reuse
          left_id:  base:d3.06
          right_id: base:d2.05
        parent: count=4 source=base depth 2 + 3, mode=reuse
          left_id:  base:d2.02
          right_id: base:d3.05
  depth 6: 25 entries
    01. id=append:fill_interior:d6.01 score=0.0210 s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.flip_v -> s1.A:single.align_row(s1.B:single) -> s0.A:single.fill_interior
        sample_count: 21/1000
        parent: count=16 source=append_right depth 5 + 1, mode=reuse
          left_id:  base:d5.01
          right_id: append:fill_interior:d1.01
        parent: count=3 source=append_right depth 3 + 3, mode=reuse
          left_id:  base:d3.06
          right_id: append:fill_interior:d3.02
        parent: count=2 source=append_right depth 1 + 5, mode=reuse
          left_id:  base:d1.11
          right_id: append:fill_interior:d5.02
    02. id=append:hollow:d6.01 score=0.0180 s0.A:single.hollow -> s0.A:single.align_row(s0.B:single) -> s0.A:single.rot90 -> s1.A:single.translate_up -> s0.A:single.translate_right -> s0.A:single.swap_colors(s0.B:single)
        sample_count: 18/1000
        parent: count=18 source=append_left depth 1 + 5, mode=reuse
          left_id:  append:hollow:d1.01
          right_id: base:d5.03
    03. id=append:hollow:d6.02 score=0.0170 s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.flip_v -> s1.A:single.align_row(s1.B:single) -> s0.A:single.hollow
        sample_count: 17/1000
        parent: count=17 source=append_right depth 5 + 1, mode=reuse
          left_id:  base:d5.01
          right_id: append:hollow:d1.01
  depth 7: 25 entries
    01. id=base:d7.01 score=0.0290 s0.A:single.align_row(s0.B:single) -> s0.A:single.rot90 -> s1.A:single.translate_up -> s0.A:single.translate_right -> s0.A:single.swap_colors(s0.B:single) -> s0.A:single.rot90 -> s1.A:single.translate_up
        sample_count: 29/1000
        parent: count=17 source=base depth 3 + 4, mode=reuse
          left_id:  base:d3.01
          right_id: base:d4.01
        parent: count=7 source=base depth 5 + 2, mode=reuse
          left_id:  base:d5.03
          right_id: base:d2.06
        parent: count=5 source=base depth 1 + 6, mode=reuse
          left_id:  base:d1.18
          right_id: base:d6.01
    02. id=append:hollow:d7.01 score=0.0230 s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.flip_v -> s1.A:single.align_row(s1.B:single) -> s0.A:single.hollow
        sample_count: 23/1000
        parent: count=17 source=append_right depth 6 + 1, mode=reuse
          left_id:  base:d6.02
          right_id: append:hollow:d1.01
        parent: count=5 source=append_right depth 1 + 6, mode=reuse
          left_id:  base:d1.11
          right_id: append:hollow:d6.02
        parent: count=1 source=append_right depth 2 + 5, mode=reuse
          left_id:  base:d2.02
          right_id: append:hollow:d5.05
    03. id=base:d7.02 score=0.0220 s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.flip_v -> s1.A:single.align_row(s1.B:single)
        sample_count: 22/1000
        parent: count=18 source=base depth 2 + 5, mode=reuse
          left_id:  base:d2.02
          right_id: base:d5.01
        parent: count=3 source=base depth 3 + 4, mode=reuse
          left_id:  base:d3.06
          right_id: base:d4.02
        parent: count=1 source=base depth 1 + 6, mode=reuse
          left_id:  base:d1.11
          right_id: base:d6.02
  depth 8: 25 entries
    01. id=append:fill_interior:d8.01 score=0.0340 s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.flip_v -> s1.A:single.align_row(s1.B:single) -> s0.A:single.fill_interior
        sample_count: 34/1000
        parent: count=17 source=append_right depth 7 + 1, mode=reuse
          left_id:  base:d7.02
          right_id: append:fill_interior:d1.01
        parent: count=6 source=append_right depth 1 + 7, mode=reuse
          left_id:  base:d1.11
          right_id: append:fill_interior:d7.02
        parent: count=6 source=append_right depth 2 + 6, mode=reuse
          left_id:  base:d2.02
          right_id: append:fill_interior:d6.01
        ... 1 more parent paths
    02. id=append:hollow:d8.01 score=0.0300 s0.A:single.align_row(s0.B:single) -> s0.A:single.rot90 -> s1.A:single.translate_up -> s0.A:single.translate_right -> s0.A:single.swap_colors(s0.B:single) -> s0.A:single.rot90 -> s1.A:single.translate_up -> s0.A:single.hollow
        sample_count: 30/1000
        parent: count=22 source=append_right depth 7 + 1, mode=reuse
          left_id:  base:d7.01
          right_id: append:hollow:d1.01
        parent: count=5 source=append_right depth 3 + 5, mode=reuse
          left_id:  base:d3.01
          right_id: append:hollow:d5.02
        parent: count=3 source=append_right depth 1 + 7, mode=reuse
          left_id:  base:d1.18
          right_id: append:hollow:d7.02
    03. id=append:hollow:d8.02 score=0.0290 s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.flip_v -> s1.A:single.align_row(s1.B:single) -> s0.A:single.hollow
        sample_count: 29/1000
        parent: count=18 source=append_right depth 7 + 1, mode=reuse
          left_id:  base:d7.02
          right_id: append:hollow:d1.01
        parent: count=5 source=append_right depth 2 + 6, mode=reuse
          left_id:  base:d2.02
          right_id: append:hollow:d6.02
        parent: count=5 source=append_right depth 3 + 5, mode=reuse
          left_id:  base:d3.06
          right_id: append:hollow:d5.05
        ... 1 more parent paths
  depth 9: 25 entries
    01. id=append:hollow:d9.01 score=0.0270 s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.flip_v -> s1.A:single.align_row(s1.B:single) -> s0.A:single.hollow
        sample_count: 27/1000
        parent: count=14 source=append_right depth 8 + 1, mode=reuse
          left_id:  base:d8.01
          right_id: append:hollow:d1.01
        parent: count=6 source=append_right depth 3 + 6, mode=reuse
          left_id:  base:d3.06
          right_id: append:hollow:d6.02
        parent: count=4 source=append_right depth 2 + 7, mode=reuse
          left_id:  base:d2.02
          right_id: append:hollow:d7.01
        ... 1 more parent paths
    02. id=base:d9.01 score=0.0260 s0.A:single.align_row(s0.B:single) -> s0.A:single.rot90 -> s1.A:single.translate_up -> s0.A:single.translate_right -> s0.A:single.swap_colors(s0.B:single) -> s0.A:single.rot90 -> s1.A:single.translate_up -> s0.A:single.rot90 -> s1.A:single.translate_up
        sample_count: 26/1000
        parent: count=12 source=base depth 7 + 2, mode=reuse
          left_id:  base:d7.01
          right_id: base:d2.06
        parent: count=7 source=base depth 1 + 8, mode=reuse
          left_id:  base:d1.18
          right_id: base:d8.02
        parent: count=7 source=base depth 3 + 6, mode=reuse
          left_id:  base:d3.01
          right_id: base:d6.08
    03. id=append:fill_interior:d9.01 score=0.0260 s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.flip_v -> s1.A:single.align_row(s1.B:single) -> s0.A:single.fill_interior
        sample_count: 26/1000
        parent: count=14 source=append_right depth 8 + 1, mode=reuse
          left_id:  base:d8.01
          right_id: append:fill_interior:d1.01
        parent: count=5 source=append_right depth 1 + 8, mode=reuse
          left_id:  base:d1.11
          right_id: append:fill_interior:d8.01
        parent: count=5 source=append_right depth 3 + 6, mode=reuse
          left_id:  base:d3.06
          right_id: append:fill_interior:d6.01
        ... 1 more parent paths
  depth 10: 25 entries
    01. id=base:d10.01 score=0.0340 s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.flip_v -> s1.A:single.align_row(s1.B:single) -> s0.A:single.translate_left -> s0.A:single.rot90 -> s1.A:single.translate_up
        sample_count: 34/1000
        parent: count=18 source=base depth 8 + 2, mode=reuse
          left_id:  base:d8.03
          right_id: base:d2.06
        parent: count=12 source=base depth 2 + 8, mode=reuse
          left_id:  base:d2.02
          right_id: base:d8.04
        parent: count=4 source=base depth 3 + 7, mode=reuse
          left_id:  base:d3.06
          right_id: base:d7.10
    02. id=base:d10.02 score=0.0310 s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.translate_down -> s0.A:single.flip_v -> s1.A:single.align_row(s1.B:single)
        sample_count: 31/1000
        parent: count=22 source=base depth 2 + 8, mode=reuse
          left_id:  base:d2.02
          right_id: base:d8.01
        parent: count=8 source=base depth 3 + 7, mode=reuse
          left_id:  base:d3.06
          right_id: base:d7.02
        parent: count=1 source=base depth 1 + 9, mode=reuse
          left_id:  base:d1.11
          right_id: base:d9.09
    03. id=append:hollow:d10.01 score=0.0290 s0.A:single.rot90 -> s1.A:single.translate_up -> s0.A:single.align_row(s0.B:single) -> s0.A:single.align_col(s0.B:single) -> s1.G:group.translate_up -> s0.A:single.translate_right -> s0.A:single.swap_colors(s0.B:single) -> s0.A:single.rot90 -> s1.A:single.translate_up -> s0.A:single.hollow
        sample_count: 29/1000
        parent: count=17 source=append_right depth 9 + 1, mode=reuse
          left_id:  base:d9.02
          right_id: append:hollow:d1.01
        parent: count=12 source=append_right depth 5 + 5, mode=reuse
          left_id:  base:d5.05
          right_id: append:hollow:d5.02
