# Research Output Issues

## Issues

| # | item | type | severity | description |
|---|---|---|---|---|
| 1 | Table_analytical_sample_and_shock_coverage_audit.xlsx and its script | table | minor | This 57-row integrated workbook and its generating script are not in the final Section 8 plan and duplicate the four separately approved coverage tables. It also retains an obsolete integrated presentation and a numbered footer, leaving the handoff inventory ambiguous about which coverage outputs are authoritative. |
| 2 | Figure_historical_conflict_and_contemporary_shock_geography.png | figure | minor | Missing shock coverage is drawn in grey in the drought and satellite-inundation panels, but the figure has no legend or in-panel key identifying grey as unavailable coverage. Readers can therefore confuse missing exposure with a low or zero shock, which is especially important for the partial satellite sample. |
| 3 | docs/AnaSOP.md Section 6.2 | plan | minor | The text refers to “Figure 3” and “Figure 4” instead of the approved output titles. This conflicts with the title-based Section 8 planning convention and creates fragile cross-references before manuscript figure numbering is assigned. |

## Severity Summary

| severity | count |
|---|---:|
| critical | 0 |
| major | 0 |
| minor | 3 |

## Recommended Next Steps

- 三个 major 问题已经解决：核心在校率结果现已覆盖替代冲突/价格定义和逐省/逐波次剔除，机制结果已加入 Holm 校正，AnaSOP 也已按最终证据更新。
- 可以继续运行 `build-content-dictionary`。在正式投稿交付前，建议再用 `figure-table-generation` 清理旧整合覆盖表、给地图补充灰色无覆盖图例，并把 AnaSOP 中的数字图号替换为输出标题。
