# Limgrave Coastal Erosion Model

## Purpose
This repository contains the simplified Python model used in the article
“Evaluating the Geomorphological Realism of Limgrave’s Southern Coast
Using Old Harry Rocks as a Real-World Analogue.”

## Model design
The model compares:
1. a uniform-bedrock control; and
2. a headland containing a structurally weakened neck.

## Erosion rules
[Explain exactly how cells or terrain are removed at each model step.]

## Parameters
[List the control and weakened-neck parameter values.]

## Separation criterion
Complete separation is recorded when the remaining connection width reaches zero.

## Running the model
1. Install the packages in requirements.txt.
2. Run:
   python limgrave_model.py

## Outputs
The script generates the terrain comparison and connection-width graph used
as Figures 11 and 12 in the article.

## Limitations
The model is dimensionless, rule-based and uncalibrated. It tests geometric
plausibility rather than reconstructing real coastal history.
