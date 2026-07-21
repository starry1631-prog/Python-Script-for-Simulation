# Limgrave Coastal Erosion Model

## Purpose

This repository contains the simplified Python model used in the article *Evaluating the Geomorphological Realism of Limgrave’s Southern Coast Using Old Harry Rocks as a Real-World Analogue*.

The model tests whether erosion concentrated along a structurally weakened headland neck can isolate a resistant offshore remnant.

## Model design

The model compares two scenarios using the same initial terrain:

1. a uniform-bedrock control; and
2. a headland containing a structurally weakened neck.

The model is a conceptual height-field simulation. Time, distance and erosion rate are dimensionless and are not calibrated to a real coastline.

## Erosion rules

At each dimensionless model step:

1. Cells at or below sea level are classified as water.
2. Wave exposure decreases exponentially with distance from the nearest water cell.
3. Erosion increases with slope and decreases with rock resistance and elevation above sea level.
4. Steep land cells near the coast undergo simplified cliff collapse by relaxing towards the mean elevation of their local neighbourhood.
5. A small fraction of the eroded material is shifted seaward and deposited in shallow water.

The erosion calculation used in the model is:

- base erosion rate: 0.60;
- slope multiplier: `1 + 0.18 × slope`, with slope limited to a maximum of 8;
- wave exposure: `exp(-distance from water / 2)`;
- elevation reduction factor: `exp(-elevation above sea level / 22)`.

Cliff collapse is applied where:

- slope is greater than 2.6;
- the cell is above sea level; and
- the cell is fewer than 14 grid cells from water.

Unstable cells move 25% towards the local neighbourhood mean elevation. Six per cent of the removed material is shifted five grid cells seaward and may be deposited in shallow water.

## Parameters

The principal model parameters are:

- grid size: 240 × 340 cells;
- model duration: 330 dimensionless steps;
- random seed: 11;
- sea level: 0;
- erosion rate: 0.60;
- collapse rate: 0.25;
- deposition fraction: 0.06.

The uniform-bedrock control uses a constant resistance of `R = 1.25`.

In the structural experiment, resistance varies spatially. The headland contains a resistant core, while resistance at the weakened neck is reduced by up to 82%. Two additional weak coastal incisions approaching the neck have resistance reductions of up to 55%. Final resistance values are restricted to the range 0.12–3.0.

## Separation criterion

The surviving connection is measured within the predefined headland-neck corridor. Complete separation is recorded when the minimum surviving bridge width reaches zero grid cells.

## Requirements

The model requires Python 3.9 or later and the packages listed in `requirements.txt`.

## Running the model

Download the repository, open a terminal in the repository folder and run:

```bash
pip install -r requirements.txt
python limgrave_coastal_isolation_with_auxiliary_chart.py
