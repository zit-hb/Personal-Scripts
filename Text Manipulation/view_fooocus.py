#!/usr/bin/env python3

# -------------------------------------------------------
# Script: view_fooocus.py
#
# Description:
# This script offers two primary functionalities:
# 1. Parsing Fooocus `log.html` files to extract metadata into a JSON file.
# 2. Serving a local web interface to browse images based on the provided JSON metadata.
#
# Usage:
# ./view_fooocus.py [command] [options]
#
# Commands:
#   - parse    Parse Fooocus `log.html` files and generate JSON metadata.
#   - serve    Start a local server to browse images using JSON metadata.
#
# Template: ubuntu22.04
#
# Requirements:
#   - BeautifulSoup4 (install via: pip install beautifulsoup4==4.12.3)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import json
import logging
import os
import re
import sys
import time
import hashlib
import socketserver
import http.server
from dataclasses import dataclass, field
from typing import List, Optional


EMBEDDED_INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Fooocus Image Search</title>
  <style>
    body {
      margin: 0;
      font-family: sans-serif;
      display: flex;
      flex-direction: column;
      height: 100vh;
    }
    header {
      background-color: #444;
      color: #fff;
      padding: 10px;
    }
    header h1 {
      margin: 0;
      font-size: 1.25rem;
    }
    #filters-container, #negative-filters-container {
      display: flex;
      flex-wrap: wrap;
      align-items: flex-end;
      gap: 8px;
      margin-top: 8px;
    }
    .filter-group {
      display: flex;
      flex-direction: column;
      font-size: 0.8rem;
    }
    .filter-group label {
      font-weight: bold;
      margin-bottom: 2px;
    }
    .filter-group input, .filter-group select {
      padding: 4px;
      font-size: 0.9rem;
      min-width: 100px;
    }
    #search-bar button {
      padding: 8px 16px;
      font-size: 0.9rem;
      background-color: #666;
      color: #fff;
      border: none;
      cursor: pointer;
      margin-left: 8px;
    }
    #search-bar button:hover {
      background-color: #888;
    }
    #main-container {
      flex: 1;
      overflow-y: auto; /* enable scrolling */
      padding: 10px;
      background-color: #f2f2f2;
    }
    .image-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(600px, 1fr));
      gap: 10px;
    }
    .image-card {
      background-color: #fff;
      border: 1px solid #ccc;
      border-radius: 4px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .image-card img {
      display: block;
      width: 100%;
      height: auto;
      object-fit: cover;
      background-color: #ddd; /* fallback if not loaded */
      cursor: pointer;
    }
    .card-content {
      padding: 8px;
      font-size: 0.85rem;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .card-content div {
      line-height: 1.3em;
    }
    .card-content strong {
      display: inline-block;
      margin-right: 4px;
    }
    .no-results {
      text-align: center;
      color: #666;
      margin-top: 20px;
    }
    #sortBar {
      margin-top: 8px;
    }
    #sortBar select {
      padding: 4px;
      font-size: 0.9rem;
      min-width: 140px;
      margin-right: 8px;
    }
  </style>
</head>
<body>

<header>
  <h1>Fooocus Image Search</h1>
  <div id="filters-container">
    <div class="filter-group">
      <label for="promptInput">Prompt</label>
      <input type="text" id="promptInput" />
    </div>
    <div class="filter-group">
      <label for="negPromptInput">Neg. Prompt</label>
      <input type="text" id="negPromptInput" />
    </div>
    <div class="filter-group">
      <label for="baseModelSelect">Base Model</label>
      <select id="baseModelSelect"></select>
    </div>
    <div class="filter-group">
      <label for="refinerModelSelect">Refiner Model</label>
      <select id="refinerModelSelect"></select>
    </div>
    <div class="filter-group">
      <label for="loraSelect">LoRA</label>
      <select id="loraSelect"></select>
    </div>
    <div class="filter-group">
      <label for="stylesSelect">Style</label>
      <select id="stylesSelect"></select>
    </div>
    <div class="filter-group">
      <label for="dirInput">Directory</label>
      <input type="text" id="dirInput" placeholder="Parent folder" />
    </div>
    <div class="filter-group">
      <label for="dateFrom">Date &gt;=</label>
      <input type="date" id="dateFrom" />
    </div>
    <div class="filter-group">
      <label for="dateTo">Date &le;</label>
      <input type="date" id="dateTo" />
    </div>
  </div>
  <div id="negative-filters-container">
    <div class="filter-group">
      <label for="notPromptInput">NOT Prompt</label>
      <input type="text" id="notPromptInput" />
    </div>
    <div class="filter-group">
      <label for="notNegPromptInput">NOT Neg. Prompt</label>
      <input type="text" id="notNegPromptInput" />
    </div>
    <div class="filter-group">
      <label for="notBaseModelSelect">NOT Base Model</label>
      <select id="notBaseModelSelect"></select>
    </div>
    <div class="filter-group">
      <label for="notRefinerModelSelect">NOT Refiner Model</label>
      <select id="notRefinerModelSelect"></select>
    </div>
    <div class="filter-group">
      <label for="notLoraSelect">NOT LoRA</label>
      <select id="notLoraSelect"></select>
    </div>
    <div class="filter-group">
      <label for="notStylesSelect">NOT Style</label>
      <select id="notStylesSelect"></select>
    </div>
    <div class="filter-group">
      <label for="notDirInput">NOT Directory</label>
      <input type="text" id="notDirInput" />
    </div>
  </div>
  <div id="sortBar">
    <label for="sortField">Sort by:</label>
    <select id="sortField">
      <option value="date">Date</option>
      <option value="prompt">Prompt</option>
      <option value="base_model">Base Model</option>
      <option value="image_dir">Directory</option>
    </select>
    <select id="sortOrder">
      <option value="desc" selected>Descending</option>
      <option value="asc">Ascending</option>
    </select>
    <button id="searchButton">Search</button>
  </div>
</header>
<div id="main-container">
  <div class="image-grid" id="imageGrid"></div>
  <div class="no-results" id="noResultsMsg" style="display: none;">No results found</div>
</div>

<script>
const DATA_URL = '/data';
const BATCH_SIZE = 100;

let allData = [];
let filteredData = [];
let currentIndex = 0;
let loading = false;

let allBaseModels = new Set();
let allRefinerModels = new Set();
let allLoraNames = new Set();
let allStyles = new Set();

async function loadData() {
  try {
    const response = await fetch(DATA_URL);
    if (!response.ok) {
      throw new Error('Failed to load data from /data');
    }
    allData = await response.json();

    // Build sets of unique values
    for (const item of allData) {
      if (item.base_model) {
        allBaseModels.add(item.base_model);
      }
      if (item.refiner_model) {
        allRefinerModels.add(item.refiner_model);
      }
      if (item.loras && item.loras.length) {
        for (const l of item.loras) {
          if (l.name) {
            allLoraNames.add(l.name);
          }
        }
      }
      if (item.styles && item.styles.length) {
        for (const s of item.styles) {
          allStyles.add(s);
        }
      }
    }

    allBaseModels = [...allBaseModels].sort();
    allRefinerModels = [...allRefinerModels].sort();
    allLoraNames = [...allLoraNames].sort();
    allStyles = [...allStyles].sort();

    populateSelect('baseModelSelect', allBaseModels);
    populateSelect('refinerModelSelect', allRefinerModels);
    populateSelect('loraSelect', allLoraNames);
    populateSelect('stylesSelect', allStyles);

    populateSelect('notBaseModelSelect', allBaseModels);
    populateSelect('notRefinerModelSelect', allRefinerModels);
    populateSelect('notLoraSelect', allLoraNames);
    populateSelect('notStylesSelect', allStyles);

    filteredData = allData;
    document.getElementById('sortField').value = 'date';
    document.getElementById('sortOrder').value = 'desc';
    sortResults('date', 'desc');
    currentIndex = 0;
    clearGrid();
    loadNextBatch();
  } catch (err) {
    console.error('Error fetching data:', err);
  }
}

function populateSelect(selectId, values) {
  const selectEl = document.getElementById(selectId);
  selectEl.innerHTML = '';
  const emptyOption = document.createElement('option');
  emptyOption.value = '';
  emptyOption.textContent = '(any)';
  selectEl.appendChild(emptyOption);

  for (const v of values) {
    const opt = document.createElement('option');
    opt.value = v;
    opt.textContent = v;
    selectEl.appendChild(opt);
  }
}

function clearGrid() {
  document.getElementById('imageGrid').innerHTML = '';
  currentIndex = 0;
}

function loadNextBatch() {
  if (loading) return;
  loading = true;

  const endIndex = Math.min(currentIndex + BATCH_SIZE, filteredData.length);
  const grid = document.getElementById('imageGrid');

  for (let i = currentIndex; i < endIndex; i++) {
    const item = filteredData[i];
    const card = createImageCard(item);
    grid.appendChild(card);
  }

  currentIndex = endIndex;
  loading = false;

  const noResultsMsg = document.getElementById('noResultsMsg');
  if (filteredData.length === 0) {
    noResultsMsg.style.display = 'block';
  } else {
    noResultsMsg.style.display = 'none';
  }
}

function createImageCard(item) {
  const card = document.createElement('div');
  card.className = 'image-card';

  const img = document.createElement('img');
  img.src = item.image_path;
  img.alt = item.prompt || 'Fooocus image';
  img.addEventListener('click', () => {
    window.open(item.image_path, '_blank');
  });

  const content = document.createElement('div');
  content.className = 'card-content';

  const promptDiv = document.createElement('div');
  promptDiv.innerHTML = `<strong>Prompt:</strong> ${item.prompt || '-'}`;

  const negPromptDiv = document.createElement('div');
  negPromptDiv.innerHTML = `<strong>Neg. Prompt:</strong> ${item.negative_prompt || '-'}`;

  const modelDiv = document.createElement('div');
  modelDiv.innerHTML = `<strong>Base Model:</strong> ${item.base_model || '-'}`;

  const refinerDiv = document.createElement('div');
  refinerDiv.innerHTML = `<strong>Refiner:</strong> ${item.refiner_model || '-'}`;

  const samplerDiv = document.createElement('div');
  samplerDiv.innerHTML = `<strong>Sampler:</strong> ${item.sampler || '-'}`;

  const schedulerDiv = document.createElement('div');
  schedulerDiv.innerHTML = `<strong>Scheduler:</strong> ${item.scheduler || '-'}`;

  const styleDiv = document.createElement('div');
  if (item.styles && item.styles.length) {
    styleDiv.innerHTML = `<strong>Styles:</strong> ${item.styles.join(', ')}`;
  } else {
    styleDiv.innerHTML = `<strong>Styles:</strong> -`;
  }

  const loraDiv = document.createElement('div');
  if (item.loras && item.loras.length) {
    const loraStrings = item.loras.map(l => `${l.name} : ${l.strength}`);
    loraDiv.innerHTML = `<strong>LoRAs:</strong> ${loraStrings.join(', ')}`;
  } else {
    loraDiv.innerHTML = `<strong>LoRAs:</strong> -`;
  }

  const resolutionDiv = document.createElement('div');
  resolutionDiv.innerHTML = `<strong>Resolution:</strong> ${item.resolution_width}x${item.resolution_height}`; 

  const guidanceScaleDiv = document.createElement('div');
  guidanceScaleDiv.innerHTML = `<strong>Guidance:</strong> ${item.guidance_scale}`;

  const sharpnessDiv = document.createElement('div');
  sharpnessDiv.innerHTML = `<strong>Sharpness:</strong> ${item.sharpness}`; 

  const directoryDiv = document.createElement('div');
  directoryDiv.innerHTML = `<strong>Directory:</strong> ${item.image_dir || '-'}`;

  const dateDiv = document.createElement('div');
  dateDiv.innerHTML = `<strong>Date:</strong> ${item.image_date || '-'}`;

  content.appendChild(promptDiv);
  content.appendChild(negPromptDiv);
  content.appendChild(modelDiv);
  content.appendChild(refinerDiv);
  content.appendChild(samplerDiv);
  content.appendChild(schedulerDiv);
  content.appendChild(styleDiv);
  content.appendChild(loraDiv);
  content.appendChild(resolutionDiv);
  content.appendChild(guidanceScaleDiv);
  content.appendChild(sharpnessDiv);
  content.appendChild(directoryDiv);
  content.appendChild(dateDiv);

  card.appendChild(img);
  card.appendChild(content);
  return card;
}

function doSearch() {
  const promptVal       = document.getElementById('promptInput').value.toLowerCase().trim();
  const negPromptVal    = document.getElementById('negPromptInput').value.toLowerCase().trim();
  const baseModelVal    = document.getElementById('baseModelSelect').value.toLowerCase().trim();
  const refinerVal      = document.getElementById('refinerModelSelect').value.toLowerCase().trim();
  const loraVal         = document.getElementById('loraSelect').value.toLowerCase().trim();
  const styleVal        = document.getElementById('stylesSelect').value.toLowerCase().trim();
  const dirVal          = document.getElementById('dirInput').value.toLowerCase().trim();
  const dateFromVal     = document.getElementById('dateFrom').value;
  const dateToVal       = document.getElementById('dateTo').value;

  const notPromptVal    = document.getElementById('notPromptInput').value.toLowerCase().trim();
  const notNegPromptVal = document.getElementById('notNegPromptInput').value.toLowerCase().trim();
  const notBaseModelVal = document.getElementById('notBaseModelSelect').value.toLowerCase().trim();
  const notRefinerVal   = document.getElementById('notRefinerModelSelect').value.toLowerCase().trim();
  const notLoraVal      = document.getElementById('notLoraSelect').value.toLowerCase().trim();
  const notStyleVal     = document.getElementById('notStylesSelect').value.toLowerCase().trim();
  const notDirVal       = document.getElementById('notDirInput').value.toLowerCase().trim();

  filteredData = allData.filter(item => {
    return (
      matchesPositive(item, {
        promptVal, negPromptVal, baseModelVal, refinerVal,
        loraVal, styleVal, dirVal, dateFromVal, dateToVal
      }) &&
      matchesNegative(item, {
        notPromptVal, notNegPromptVal, notBaseModelVal,
        notRefinerVal, notLoraVal, notStyleVal, notDirVal
      })
    );
  });

  const sortField = document.getElementById('sortField').value;
  const sortOrder = document.getElementById('sortOrder').value;
  sortResults(sortField, sortOrder);

  clearGrid();
  loadNextBatch();
}

function matchesPositive(item, {
  promptVal, negPromptVal, baseModelVal, refinerVal,
  loraVal, styleVal, dirVal, dateFromVal, dateToVal
}) {
  if (promptVal && (!item.prompt || !item.prompt.toLowerCase().includes(promptVal))) {
    return false;
  }
  if (negPromptVal && (!item.negative_prompt || !item.negative_prompt.toLowerCase().includes(negPromptVal))) {
    return false;
  }
  if (baseModelVal && (!item.base_model || item.base_model.toLowerCase() !== baseModelVal)) {
    return false;
  }
  if (refinerVal && (!item.refiner_model || item.refiner_model.toLowerCase() !== refinerVal)) {
    return false;
  }
  if (loraVal) {
    if (!item.loras || !item.loras.some(l => l.name && l.name.toLowerCase() === loraVal)) {
      return false;
    }
  }
  if (styleVal) {
    if (!item.styles || !item.styles.some(s => s.toLowerCase() === styleVal)) {
      return false;
    }
  }
  if (dirVal && (!item.image_dir || !item.image_dir.toLowerCase().includes(dirVal))) {
    return false;
  }

  if (dateFromVal || dateToVal) {
    // item.image_date is e.g. '2025-01-03 12:34:56', so substring
    const itemDate = (item.image_date || '').slice(0, 10);
    if (dateFromVal && itemDate < dateFromVal) {
      return false;
    }
    if (dateToVal && itemDate > dateToVal) {
      return false;
    }
  }

  return true;
}

function matchesNegative(item, {
  notPromptVal, notNegPromptVal, notBaseModelVal,
  notRefinerVal, notLoraVal, notStyleVal, notDirVal
}) {
  if (notPromptVal && item.prompt && item.prompt.toLowerCase().includes(notPromptVal)) {
    return false;
  }
  if (notNegPromptVal && item.negative_prompt && item.negative_prompt.toLowerCase().includes(notNegPromptVal)) {
    return false;
  }
  if (notBaseModelVal && item.base_model && item.base_model.toLowerCase() === notBaseModelVal) {
    return false;
  }
  if (notRefinerVal && item.refiner_model && item.refiner_model.toLowerCase() === notRefinerVal) {
    return false;
  }
  if (notLoraVal && item.loras && item.loras.some(l => l.name && l.name.toLowerCase() === notLoraVal)) {
    return false;
  }
  if (notStyleVal && item.styles && item.styles.some(s => s.toLowerCase() === notStyleVal)) {
    return false;
  }
  if (notDirVal && item.image_dir && item.image_dir.toLowerCase().includes(notDirVal)) {
    return false;
  }
  return true;
}

function sortResults(field, order) {
  filteredData.sort((a, b) => {
    let valA, valB;
    switch (field) {
      case 'date':
        valA = new Date(a.image_date || '').getTime();
        valB = new Date(b.image_date || '').getTime();
        break;
      case 'prompt':
        valA = a.prompt || '';
        valB = b.prompt || '';
        break;
      case 'base_model':
        valA = a.base_model || '';
        valB = b.base_model || '';
        break;
      case 'image_dir':
        valA = a.image_dir || '';
        valB = b.image_dir || '';
        break;
      default:
        valA = '';
        valB = '';
    }
    if (valA < valB) return order === 'asc' ? -1 : 1;
    if (valA > valB) return order === 'asc' ? 1 : -1;
    return 0;
  });
}

function handleScroll() {
  const mainContainer = document.getElementById('main-container');
  const { scrollTop, scrollHeight, clientHeight } = mainContainer;
  const nearBottom = (scrollTop + clientHeight) >= (scrollHeight - 10);

  if (nearBottom && currentIndex < filteredData.length) {
    loadNextBatch();
  }
}

document.addEventListener('DOMContentLoaded', () => {
  loadData();
  document.getElementById('searchButton').addEventListener('click', doSearch);

  const allInputs = document.querySelectorAll(
    '#filters-container input, #filters-container select,' +
    '#negative-filters-container input, #negative-filters-container select,' +
    '#sortBar select'
  );
  allInputs.forEach(el => {
    el.addEventListener('keypress', e => {
      if (e.key === 'Enter') {
        doSearch();
      }
    });
  });

  document.getElementById('main-container').addEventListener('scroll', handleScroll);
});
</script>
</body>
</html>
"""


class FooocusHandler(http.server.SimpleHTTPRequestHandler):
    """
    Custom handler to securely serve:
      1) "/" -> embedded index HTML
      2) "/data" -> the JSON data in memory
      3) "/images/<hash>" -> whitelisted local images
      4) anything else -> 404
    """

    def __init__(self, *args, **kwargs):
        self.server_json = kwargs.pop("server_json", [])
        self.path_map = kwargs.pop("path_map", {})
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == "/":
            self.serve_index()
            return

        if self.path == "/data":
            self.serve_data()
            return

        if self.path.startswith("/images/"):
            self.serve_image()
            return

        self.send_error(404, "Not Found")

    def serve_index(self):
        try:
            content = EMBEDDED_INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Failed to serve index: {e}")

    def serve_data(self):
        try:
            data_bytes = json.dumps(self.server_json).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(data_bytes)
        except Exception as e:
            self.send_error(500, f"Failed to generate JSON: {e}")

    def serve_image(self):
        parts = self.path.split("/")
        if len(parts) != 3:
            self.send_error(404, "Invalid image path")
            return

        image_hash = parts[2]
        if image_hash not in self.path_map:
            self.send_error(404, "Image hash not found")
            return

        local_path = self.path_map[image_hash]
        if not os.path.isfile(local_path):
            self.send_error(404, "Image file missing on disk")
            return

        _, ext = os.path.splitext(local_path)
        if ext.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"):
            content_type = f"image/{ext.lower().strip('.')}"
        else:
            content_type = "application/octet-stream"

        try:
            with open(local_path, "rb") as f:
                image_data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(image_data)
        except Exception as e:
            self.send_error(500, f"Failed to read image file: {e}")


@dataclass
class FooocusImageData:
    """Data class to store metadata for a single Fooocus-generated image."""

    image_path: str
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    fooocus_v2_expansion: Optional[str] = None
    styles: Optional[List[str]] = None
    performance: Optional[str] = None
    resolution_width: Optional[int] = None
    resolution_height: Optional[int] = None
    sharpness: Optional[float] = None
    guidance_scale: Optional[float] = None
    adm_guidance: Optional[List[float]] = None
    base_model: Optional[str] = None
    refiner_model: Optional[str] = None
    refiner_switch: Optional[float] = None
    sampler: Optional[str] = None
    scheduler: Optional[str] = None
    seed: Optional[str] = None
    version: Optional[str] = None
    loras: List[dict] = field(default_factory=list)
    image_dir: Optional[str] = None
    image_date: Optional[str] = None


def parse_resolution(resolution_str: str):
    """
    Parses a resolution string of the form '(2560, 1536)' into (width, height).
    Returns (None, None) if parsing fails.
    """
    match = re.match(r"\((\d+),\s*(\d+)\)", resolution_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def parse_float_tuple(tuple_str: str):
    """
    Parses a string like '(1.5, 0.8, 0.3)' into a float list.
    Returns None if parsing fails.
    """
    import ast

    try:
        parsed = ast.literal_eval(tuple_str)
        if isinstance(parsed, tuple):
            return [float(x) for x in parsed]
        if isinstance(parsed, (float, int)):
            return [float(parsed)]
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except Exception:
        pass
    return None


def parse_styles(styles_str: str):
    """
    Parses a string like "['MRE Space Art', 'MRE Manga']" into a list of strings.
    Returns None if parsing fails.
    """
    import ast

    try:
        parsed = ast.literal_eval(styles_str)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        if isinstance(parsed, str):
            return [parsed]
    except Exception:
        pass
    return None


def parse_image_metadata(image_div, base_dir: str) -> Optional[FooocusImageData]:
    """
    Given a BeautifulSoup <div class="image-container">, parse out the metadata
    and return a FooocusImageData instance. Returns None if the image file doesn't exist.
    """
    link_tag = image_div.find("a")
    if not link_tag or "href" not in link_tag.attrs:
        return None

    image_rel_path = link_tag["href"]
    image_abs_path = os.path.abspath(os.path.join(base_dir, image_rel_path))

    if not os.path.isfile(image_abs_path):
        return None

    metadata_obj = FooocusImageData(image_path=image_abs_path)
    metadata_obj.image_dir = os.path.basename(os.path.dirname(image_abs_path))
    stat_info = os.stat(image_abs_path)
    ts = stat_info.st_mtime
    metadata_obj.image_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

    metadata_table = image_div.find("table", class_="metadata")
    if not metadata_table:
        return metadata_obj

    rows = metadata_table.find_all("tr")
    for row in rows:
        key_td = row.find("td", class_="label")
        value_td = row.find("td", class_="value")
        if not key_td or not value_td:
            continue

        key = key_td.get_text(strip=True)
        value = value_td.get_text(strip=True)

        if key == "Prompt":
            metadata_obj.prompt = value
        elif key == "Negative Prompt":
            metadata_obj.negative_prompt = value
        elif key == "Fooocus V2 Expansion":
            metadata_obj.fooocus_v2_expansion = value
        elif key == "Styles":
            metadata_obj.styles = parse_styles(value)
        elif key == "Performance":
            metadata_obj.performance = value
        elif key == "Resolution":
            w, h = parse_resolution(value)
            metadata_obj.resolution_width = w
            metadata_obj.resolution_height = h
        elif key == "Sharpness":
            try:
                metadata_obj.sharpness = float(value)
            except ValueError:
                pass
        elif key == "Guidance Scale":
            try:
                metadata_obj.guidance_scale = float(value)
            except ValueError:
                pass
        elif key == "ADM Guidance":
            metadata_obj.adm_guidance = parse_float_tuple(value)
        elif key == "Base Model":
            metadata_obj.base_model = value
        elif key == "Refiner Model":
            metadata_obj.refiner_model = value
        elif key == "Refiner Switch":
            try:
                metadata_obj.refiner_switch = float(value)
            except ValueError:
                pass
        elif key == "Sampler":
            metadata_obj.sampler = value
        elif key == "Scheduler":
            metadata_obj.scheduler = value
        elif key == "Seed":
            metadata_obj.seed = value
        elif key == "Version":
            metadata_obj.version = value
        elif key.lower().startswith("lora "):
            parts = value.split(":", 1)
            lora_name = parts[0].strip()
            lora_strength = parts[1].strip() if len(parts) > 1 else ""
            metadata_obj.loras.append({"name": lora_name, "strength": lora_strength})

    return metadata_obj


def find_index_files(root_path: str, recursive: bool):
    """
    Finds all 'log.html' files in the specified path.
    If 'recursive' is False, only the top-level directory is scanned.
    """
    index_files = []

    if not os.path.isdir(root_path):
        logging.error(f"Path '{root_path}' is not a valid directory.")
        sys.exit(1)

    if recursive:
        for dir_path, _, filenames in os.walk(root_path):
            if "log.html" in filenames:
                index_files.append(os.path.join(dir_path, "log.html"))
    else:
        candidate = os.path.join(root_path, "log.html")
        if os.path.isfile(candidate):
            index_files.append(candidate)

    if not index_files:
        logging.warning("No log.html files found in the specified path.")
    else:
        logging.info(f"Found {len(index_files)} log.html file(s).")

    return index_files


def parse_index_file(index_file: str):
    """
    Parses a single log.html file for Fooocus metadata.
    Returns a list of FooocusImageData objects.
    """
    logging.debug(f"Parsing index file: {index_file}")
    result = []

    # We only import BeautifulSoup here (lazy import), so
    # if the user isn't running 'parse', we don't need it installed.
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logging.error("BeautifulSoup is not installed")
        sys.exit(1)

    with open(index_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    image_divs = soup.find_all("div", class_="image-container")
    base_dir = os.path.dirname(index_file)

    for div in image_divs:
        metadata_obj = parse_image_metadata(div, base_dir)
        if metadata_obj:
            result.append(metadata_obj)

    logging.debug(f"Found {len(result)} valid image entries in {index_file}.")
    return result


def do_parse(args):
    """
    Implements the 'parse' subcommand.
    Recursively (or not) searches for Fooocus log.html files, parses them, and
    writes the metadata to a JSON file.
    """
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    index_files = find_index_files(args.path, args.recursive)

    all_metadata = []
    for index_file in index_files:
        file_entries = parse_index_file(index_file)
        all_metadata.extend(file_entries)

    if not all_metadata:
        logging.warning("No valid image metadata found.")
        return

    dicts_for_json = []
    for item in all_metadata:
        dicts_for_json.append(
            {
                "image_path": item.image_path,
                "prompt": item.prompt,
                "negative_prompt": item.negative_prompt,
                "fooocus_v2_expansion": item.fooocus_v2_expansion,
                "styles": item.styles,
                "performance": item.performance,
                "resolution_width": item.resolution_width,
                "resolution_height": item.resolution_height,
                "sharpness": item.sharpness,
                "guidance_scale": item.guidance_scale,
                "adm_guidance": item.adm_guidance,
                "base_model": item.base_model,
                "refiner_model": item.refiner_model,
                "refiner_switch": item.refiner_switch,
                "sampler": item.sampler,
                "scheduler": item.scheduler,
                "seed": item.seed,
                "version": item.version,
                "loras": item.loras,
                "image_dir": item.image_dir,
                "image_date": item.image_date,
            }
        )

    output_path = os.path.abspath(args.output)
    try:
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(dicts_for_json, out_f, indent=2, ensure_ascii=False)
        logging.info(f"Metadata written to '{output_path}'")
    except Exception as e:
        logging.error(f"Could not write to JSON file '{output_path}': {e}")


def hashed_path(path: str) -> str:
    """
    Create a stable hash for the file path so it can be
    referred to by /images/<hash> instead of the absolute path.
    """
    normed_path = os.path.normcase(os.path.abspath(path))
    return hashlib.md5(normed_path.encode("utf-8")).hexdigest()


def do_serve(args):
    """
    Implements the 'serve' subcommand.
    Reads a JSON file with metadata and starts a local web server to browse it.
    """
    json_file = os.path.abspath(args.input)
    if not os.path.isfile(json_file):
        print(f"Error: JSON file '{json_file}' does not exist.")
        sys.exit(1)

    with open(json_file, "r", encoding="utf-8") as f:
        fooocus_data = json.load(f)

    path_map = {}
    server_json = []

    for item in fooocus_data:
        original_path = item.get("image_path", "")
        h = hashed_path(original_path)
        path_map[h] = original_path

        new_item = dict(item)
        new_item["image_path"] = f"/images/{h}"
        server_json.append(new_item)

    host = args.host
    port = args.port

    # We set allow_reuse_address to True to avoid a port-block issue.
    class CustomTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    print(f"Serving Fooocus data at http://{host}:{port}")
    print("Press Ctrl+C to stop.")

    handler_args = {"server_json": server_json, "path_map": path_map}

    with CustomTCPServer(
        (host, port), lambda *a, **kw: FooocusHandler(*a, **{**kw, **handler_args})
    ) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            print("\nShutting down server...")
            httpd.server_close()


def main():
    parser = argparse.ArgumentParser(
        description="Parse Fooocus log.html files or serve a local web interface to browse them."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: parse
    parse_parser = subparsers.add_parser(
        "parse", help="Parse Fooocus log.html files to produce JSON metadata."
    )
    parse_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="fooocus_data.json",
        help="Output JSON file for metadata.",
    )
    parse_parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Process directories recursively.",
    )
    parse_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    parse_parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="The directory to search for log.html files.",
    )

    # Subcommand: serve
    serve_parser = subparsers.add_parser(
        "serve", help="Serve the JSON metadata via a local web interface."
    )
    serve_parser.add_argument(
        "-H",
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host IP to bind to.",
    )
    serve_parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8001,
        help="Port to serve on.",
    )
    serve_parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="fooocus_data.json",
        help="JSON file with metadata.",
    )

    args = parser.parse_args()

    if args.command == "parse":
        do_parse(args)
    elif args.command == "serve":
        do_serve(args)


if __name__ == "__main__":
    main()
