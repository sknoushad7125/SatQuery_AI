# OPTICAL-SAR LABEL MAPPING

## Dataset: SEN12MS

SEN12MS ground-truth labels are derived from MODIS Land Cover (IGBP classification). The IGBP system contains 17 classes.

Our model (`FeatureLevelOpticalSARFusion`) predicts 4 simplified downstream classes:
1. `built-up area`
2. `water body`
3. `vegetation`
4. `bare land`

### Justified Scientific Mapping
To train the 4-class classifier using the SEN12MS MODIS labels, the following mapping is enforced during preprocessing:

| MODIS (IGBP) Index | Original Class Description | Mapped Model Class | Justification |
| :--- | :--- | :--- | :--- |
| 1-5, 8, 9 | Forests & Savannas | `vegetation` | Tree canopy and woodland dominance perfectly align with broad vegetation. |
| 10 | Grasslands | `vegetation` | Dense grass canopy triggers high NDVI/NIR response similar to forest. |
| 12 | Croplands | `vegetation` | Active agricultural biomass. |
| 13 | Urban and Built-up | `built-up area` | Direct structural mapping; high SAR backscatter (double-bounce). |
| 16 | Barren | `bare land` | Soil, rock, sand; minimal biomass. |
| 17 | Water Bodies | `water body` | Direct mapping; low SAR backscatter (specular reflection). |
| 6, 7, 11 | Shrublands/Wetlands | *EXCLUDED* | Mixed signatures. Too ambiguous for strict 4-class categorization. |
| 14, 15 | Snow and Ice | *EXCLUDED* | Not relevant to the primary 4-class taxonomy required by the user context. |

This mapping explicitly avoids arbitrary generalizations, safely excluding mixed classes (like wetlands) to prevent polluting the classification head with ambiguous optical/SAR features.
