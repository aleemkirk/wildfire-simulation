# Wildfire Data Directory

This directory contains datasets for training and evaluating the wildfire spread prediction model.

## Directory Structure

### `raw/`
Contains raw, unprocessed wildfire datasets from various sources.

### `processed/`
Contains preprocessed data ready for model training and evaluation.

## Data Sources

### 1. NASA FIRMS (Fire Information for Resource Management System)
- **URL**: https://firms.modaps.eosdis.nasa.gov/
- **Description**: Near real-time wildfire detection data from MODIS and VIIRS satellites
- **Data includes**: Fire locations, brightness, confidence levels, acquisition times

### 2. Kaggle Wildfire Datasets
- **URL**: https://www.kaggle.com/datasets (search for "wildfire")
- **Description**: Historical wildfire records and preprocessed datasets
- **Data includes**: Fire perimeters, burn areas, dates, locations

### 3. LANDFIRE
- **URL**: https://landfire.gov/
- **Description**: Vegetation and fuel data for the United States
- **Data includes**: Fuel types, vegetation density, fuel moisture content

### 4. NOAA Weather Data
- **URL**: https://www.ncdc.noaa.gov/
- **Description**: Historical weather records
- **Data includes**: Wind speed/direction, temperature, humidity, precipitation

## Data Format

Processed data should be in one of the following formats:
- **NumPy arrays** (.npy): For grid-based spatial data
- **Pandas DataFrames** (.csv, .parquet): For tabular data
- **GeoJSON/Shapefiles**: For geospatial vector data (optional)

## Usage

1. Download raw data from sources listed above
2. Place raw data files in `raw/` directory
3. Run preprocessing scripts from `src/preprocessing/` to generate processed data
4. Processed data will be saved to `processed/` directory

## Notes

- Do not commit large data files to git repository
- Add data files to `.gitignore` if necessary
- Document any custom datasets added to this directory