#!/usr/bin/env python3

# -------------------------------------------------------
# Script: map_edit_images.py
#
# Description:
# Starts a small Flask web application that shows a Leaflet
# map with every image in the chosen directory rendered as a
# miniature thumbnail at its GPS position.
#
# Images can be uploaded from the browser (they are written
# into the same directory and appear on the map right away)
# and the GPS position of an image can be corrected either by
# dragging its marker or by typing coordinates. The EXIF GPS
# block of the file on disk is rewritten in place; all other
# image data and metadata stays untouched.
#
# Usage:
#   ./map_edit_images.py [options] directory
#
# Arguments:
#   - [directory]: Path to the directory containing images with metadata.
#
# Options:
#   -p, --port PORT         Port to run the server on (default: 5000).
#   -H, --host HOST         Host to run the server on (default: 0.0.0.0).
#   -t, --title TITLE       Page title (default: Hendrik's Image Map).
#   -q, --quality QUALITY   JPEG quality for the *full* images (default: 85).
#   -TW, --thumb-width PX   Thumbnail width in pixels (default: 600).
#   -TH, --thumb-height PX  Thumbnail height in pixels (default: 400).
#   -L, --locate            Enable browser geolocation button on the map.
#   -M, --max-upload MB     Maximum size of a single upload request (default: 128).
#   -R, --read-only         Disable uploading and coordinate editing.
#   -v, --verbose           Enable verbose logging (INFO level).
#   -vv, --debug            Enable debug logging (DEBUG level).
#
# Template: ubuntu24.04
#
# Requirements:
#   - Flask (install via: pip install flask==3.1.0)
#   - Pillow (install via: pip install Pillow==11.1.0)
#   - piexif (install via: pip install piexif==1.1.3)
#
# Note:
#   There is no authentication. Anybody who can reach the port can
#   write files into the image directory and change coordinates.
#   Bind to 127.0.0.1 if that is not what you want.
#
# -------------------------------------------------------
# © 2026 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import copy
import hashlib
import io
import logging
import os
import shutil
import threading
from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Optional, Tuple

from flask import Flask, Response, abort, jsonify, render_template_string, request
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import piexif


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg")

TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{ page_title }}</title>

  <!-- Leaflet core -->
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha384-sHL9NAb7lN7rfvG5lfHpm643Xkcjzp4jFvuavGOndn6pjVqS6ny56CAt3nsEVT4H"
    crossorigin="anonymous"
  />

  <!-- Marker-cluster plugin -->
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"
    integrity="sha384-pmjIAcz2bAn0xukfxADbZIb3t8oRT9Sv0rvO+BR5Csr6Dhqq+nZs59P0pPKQJkEV"
    crossorigin="anonymous"
  />
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"
    integrity="sha384-wgw+aLYNQ7dlhK47ZPK7FRACiq7ROZwgFNg0m04avm4CaXS+Z9Y7nMu8yNjBKYC+"
    crossorigin="anonymous"
  />

  <style>
    :root {
      --bg: #ffffff;
      --fg: #14181f;
      --muted: #6b7280;
      --line: #e3e6ea;
      --accent: #1f6feb;
      --accent-fg: #ffffff;
      --warn: #b42318;
      --radius: 10px;
      --shadow: 0 8px 28px rgba(15, 23, 42, .18);
      --font: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }

    html, body { height: 100%; margin: 0; }
    body { font-family: var(--font); color: var(--fg); }
    #map { height: 100%; width: 100%; position: relative; }
    [hidden] { display: none !important; }

    button, input, label { font: inherit; }
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      padding: 7px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--fg);
      font-size: 13px;
      line-height: 1.2;
      cursor: pointer;
      white-space: nowrap;
    }
    .btn:hover { border-color: #c9ced6; }
    .btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
    .btn.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: var(--accent-fg);
      font-weight: 600;
    }
    .btn.primary:hover { background: #1a5fd0; }
    .btn[disabled] { opacity: .55; cursor: default; }

    /* ---------- side panel ---------- */
    .panel {
      position: absolute;
      top: 10px;
      right: 10px;
      z-index: 1000;
      width: 290px;
      max-width: calc(100vw - 20px);
      max-height: calc(100% - 20px);
      display: flex;
      flex-direction: column;
      background: var(--bg);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
    }
    .panel-title {
      font-size: 13px;
      font-weight: 600;
      letter-spacing: .01em;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .panel-toggle {
      border: none;
      background: none;
      color: var(--muted);
      font-size: 18px;
      line-height: 1;
      padding: 2px 6px;
      cursor: pointer;
      border-radius: 6px;
    }
    .panel-toggle:hover { background: #f2f4f7; }
    .panel-body {
      padding: 12px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .panel.collapsed .panel-body { display: none; }

    .row { display: flex; gap: 8px; flex-wrap: wrap; }
    .hint { margin: 0; font-size: 12px; color: var(--muted); line-height: 1.45; }
    .status {
      font-size: 12px;
      line-height: 1.45;
      padding: 8px 10px;
      border-radius: 8px;
      background: #f2f5f9;
      color: #33405a;
    }
    .status.error { background: #fdf1f0; color: var(--warn); }
    .section-title {
      margin: 4px 0 0;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: .06em;
      color: var(--muted);
    }
    .divider { height: 1px; background: var(--line); margin: 2px 0; }

    .thumb-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .thumb-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0;
      background: #fff;
      cursor: pointer;
      overflow: hidden;
      text-align: left;
    }
    .thumb-card img {
      display: block;
      width: 100%;
      height: 72px;
      object-fit: cover;
      background: #f2f4f7;
    }
    .thumb-card .name {
      display: block;
      padding: 5px 6px;
      font-size: 11px;
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .thumb-card:hover { border-color: var(--accent); }
    .thumb-card.active { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(31,111,235,.2); }

    .count-summary { font-size: 12px; color: var(--muted); }

    /* ---------- popup ---------- */
    .thumb-popup img.thumb {
      display: block;
      border-radius: 6px;
      cursor: zoom-in;
      max-width: 100%;
      height: auto;
    }
    .popup-name {
      margin-top: 8px;
      font-size: 12px;
      font-weight: 600;
      word-break: break-all;
    }
    .popup-form {
      margin-top: 8px;
      display: grid;
      grid-template-columns: 1fr 1fr auto;
      gap: 6px;
      align-items: end;
    }
    .field { display: flex; flex-direction: column; gap: 3px; }
    .field span {
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: .06em;
      color: var(--muted);
    }
    .field input {
      width: 100%;
      box-sizing: border-box;
      padding: 5px 7px;
      border: 1px solid var(--line);
      border-radius: 6px;
      font-family: var(--mono);
      font-size: 12px;
    }
    .field input:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
    .popup-msg { margin-top: 6px; font-size: 11px; color: var(--muted); }
    .popup-msg.error { color: var(--warn); }
    .leaflet-popup-content { margin: 12px; }

    /* ---------- drop overlay & placing banner ---------- */
    #drop-overlay {
      position: absolute;
      inset: 0;
      z-index: 1200;
      display: none;
      align-items: center;
      justify-content: center;
      background: rgba(20, 24, 31, .45);
      backdrop-filter: blur(1px);
    }
    #drop-overlay.visible { display: flex; }
    .drop-inner {
      padding: 18px 26px;
      border: 2px dashed #fff;
      border-radius: 12px;
      color: #fff;
      font-size: 15px;
      font-weight: 600;
    }

    #place-banner {
      position: absolute;
      left: 50%;
      bottom: 22px;
      transform: translateX(-50%);
      z-index: 1100;
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 9px 12px;
      border-radius: 999px;
      background: var(--fg);
      color: #fff;
      font-size: 13px;
      box-shadow: var(--shadow);
      max-width: calc(100vw - 40px);
    }
    #place-banner .btn { background: transparent; border-color: #4a5262; color: #fff; padding: 4px 10px; }
    #place-banner .btn:hover { border-color: #7c8393; }
    .leaflet-container.placing { cursor: crosshair; }

    @media (max-width: 640px) {
      .panel { width: calc(100vw - 20px); }
    }
    @media (prefers-reduced-motion: reduce) {
      * { transition: none !important; animation: none !important; }
    }
  </style>
</head>
<body>
  <div id="map"></div>

  <div id="panel" class="panel">
    <header class="panel-head">
      <span class="panel-title">{{ page_title }}</span>
      <button id="panel-toggle" class="panel-toggle" title="Collapse panel" aria-label="Collapse panel">&minus;</button>
    </header>
    <div class="panel-body">
      {% if editable %}
      <div class="row">
        <label class="btn primary">
          Add photos
          <input id="file-input" type="file" accept="image/jpeg,.jpg,.jpeg" multiple hidden />
        </label>
        {% if enable_geolocate %}
        <button id="locate-btn" class="btn" title="Show my location">Locate me</button>
        {% endif %}
      </div>
      <p class="hint">Drop JPEGs anywhere on the map to add them. Drag a marker, or type coordinates in its popup, to correct a position.</p>
      {% elif enable_geolocate %}
      <div class="row">
        <button id="locate-btn" class="btn" title="Show my location">Locate me</button>
      </div>
      {% endif %}

      <div id="status" class="status" hidden></div>

      <div id="unplaced-section" hidden>
        <div class="divider"></div>
        <h2 class="section-title">Without coordinates (<span id="unplaced-count">0</span>)</h2>
        <p class="hint">Choose a photo, then click the map to place it.</p>
        <div id="unplaced-list" class="thumb-grid"></div>
      </div>

      <div class="divider"></div>
      <span id="count-summary" class="count-summary"></span>
    </div>
  </div>

  <div id="drop-overlay"><div class="drop-inner">Drop JPEGs to add them</div></div>

  <div id="place-banner" hidden>
    <span id="place-text"></span>
    <button id="place-cancel" class="btn">Cancel</button>
  </div>

  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha384-cxOPjt7s7Iz04uaHJceBmS+qpjv2JkIHNVcuOrM+YHwZOmJGBXI00mdUXEq65HTH"
    crossorigin="anonymous">
  </script>
  <script
    src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster-src.js"
    integrity="sha384-xLgzMQOvDhPE6lQoFpJJOFU2aMYsKD5eSSt9q3aR1RREx3Y+XsnqtSDZd+PhAcob"
    crossorigin="anonymous">
  </script>

  <script>
  (function () {
    'use strict';

    const EDITABLE = {{ 'true' if editable else 'false' }};
    const POLL_MS = 5000;

    const map = L.map('map');
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 21,
      attribution: '© OpenStreetMap'
    }).addTo(map);

    const cluster = L.markerClusterGroup({
      spiderfyOnMaxZoom: true,
      showCoverageOnHover: false,
      maxClusterRadius: 80
    });
    map.addLayer(cluster);

    const state = {
      version: -1,
      images: new Map(),
      markers: new Map(),
      placingId: null,
      dragging: null,
      didInitialFit: false
    };

    const el = {
      panel: document.getElementById('panel'),
      panelToggle: document.getElementById('panel-toggle'),
      fileInput: document.getElementById('file-input'),
      status: document.getElementById('status'),
      unplacedSection: document.getElementById('unplaced-section'),
      unplacedList: document.getElementById('unplaced-list'),
      unplacedCount: document.getElementById('unplaced-count'),
      summary: document.getElementById('count-summary'),
      dropOverlay: document.getElementById('drop-overlay'),
      placeBanner: document.getElementById('place-banner'),
      placeText: document.getElementById('place-text'),
      placeCancel: document.getElementById('place-cancel'),
      locateBtn: document.getElementById('locate-btn')
    };

    L.DomEvent.disableClickPropagation(el.panel);
    L.DomEvent.disableScrollPropagation(el.panel);
    L.DomEvent.disableClickPropagation(el.placeBanner);

    el.panelToggle.addEventListener('click', function () {
      const collapsed = el.panel.classList.toggle('collapsed');
      el.panelToggle.innerHTML = collapsed ? '+' : '&minus;';
      el.panelToggle.title = collapsed ? 'Expand panel' : 'Collapse panel';
    });

    let statusTimer = null;
    function setStatus(text, isError) {
      if (statusTimer) { clearTimeout(statusTimer); statusTimer = null; }
      if (!text) {
        el.status.hidden = true;
        el.status.textContent = '';
        return;
      }
      el.status.hidden = false;
      el.status.textContent = text;
      el.status.classList.toggle('error', Boolean(isError));
      if (!isError) {
        statusTimer = setTimeout(function () { setStatus(''); }, 6000);
      }
    }

    function fmt(value) {
      return Number(value).toFixed(6);
    }

    /* ---------- server calls ---------- */

    async function readJson(res) {
      let data = null;
      try { data = await res.json(); } catch (err) { data = null; }
      if (!res.ok) {
        throw new Error((data && data.error) || ('Request failed with status ' + res.status));
      }
      return data;
    }

    async function fetchImages() {
      return readJson(await fetch('/api/images', { cache: 'no-store' }));
    }

    async function saveLocation(id, lat, lon) {
      const res = await fetch('/api/images/' + encodeURIComponent(id) + '/location', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lat: lat, lon: lon })
      });
      const data = await readJson(res);
      state.version = -1;
      await refresh();
      return data.image;
    }

    async function uploadFiles(fileList) {
      const files = Array.prototype.slice.call(fileList).filter(function (f) {
        return /\.jpe?g$/i.test(f.name);
      });
      if (!files.length) {
        setStatus('Only JPEG files can be added.', true);
        return;
      }
      const form = new FormData();
      files.forEach(function (f) { form.append('files', f, f.name); });
      setStatus('Adding ' + files.length + ' file' + (files.length === 1 ? '' : 's') + '…');
      try {
        const data = await readJson(await fetch('/api/upload', { method: 'POST', body: form }));
        state.version = -1;
        await refresh();
        reportUpload(data);
        focusOn(data.added || []);
      } catch (err) {
        setStatus(err.message, true);
      }
    }

    function reportUpload(data) {
      const added = (data.added || []).length;
      const skipped = data.skipped || [];
      const parts = [];
      parts.push(added + ' photo' + (added === 1 ? '' : 's') + ' added');
      if (skipped.length) {
        parts.push(skipped.length + ' skipped: ' + skipped.map(function (s) {
          return s.name + ' (' + s.reason + ')';
        }).join(', '));
      }
      setStatus(parts.join(' · '), added === 0);
    }

    function focusOn(added) {
      const located = added.filter(function (img) { return img.lat !== null && img.lon !== null; });
      if (!located.length) return;
      const b = L.latLngBounds(located.map(function (img) { return [img.lat, img.lon]; }));
      map.fitBounds(b.pad(0.3), { maxZoom: 16 });
    }

    /* ---------- markers ---------- */

    function popupContent(id) {
      const img = state.images.get(id);
      const root = document.createElement('div');
      root.className = 'thumb-popup';

      const picture = document.createElement('img');
      picture.className = 'thumb';
      picture.src = '/thumbnail/' + encodeURIComponent(id);
      picture.width = img.thumb_w;
      picture.height = img.thumb_h;
      picture.alt = img.filename;
      picture.title = 'Open the full image';
      picture.addEventListener('click', function () {
        window.open('/image/' + encodeURIComponent(id), '_blank', 'noopener');
      });
      root.appendChild(picture);

      const name = document.createElement('div');
      name.className = 'popup-name';
      name.textContent = img.filename;
      root.appendChild(name);

      if (!EDITABLE) {
        const coords = document.createElement('div');
        coords.className = 'popup-msg';
        coords.textContent = fmt(img.lat) + ', ' + fmt(img.lon);
        root.appendChild(coords);
        return root;
      }

      const form = document.createElement('div');
      form.className = 'popup-form';

      function field(labelText, value) {
        const wrap = document.createElement('label');
        wrap.className = 'field';
        const span = document.createElement('span');
        span.textContent = labelText;
        const input = document.createElement('input');
        input.type = 'number';
        input.step = 'any';
        input.value = fmt(value);
        wrap.appendChild(span);
        wrap.appendChild(input);
        form.appendChild(wrap);
        return input;
      }

      const latInput = field('Latitude', img.lat);
      const lonInput = field('Longitude', img.lon);

      const save = document.createElement('button');
      save.className = 'btn primary';
      save.textContent = 'Save';
      form.appendChild(save);
      root.appendChild(form);

      const msg = document.createElement('div');
      msg.className = 'popup-msg';
      msg.textContent = 'Or drag the marker to move this photo.';
      root.appendChild(msg);

      save.addEventListener('click', async function () {
        const lat = parseFloat(latInput.value);
        const lon = parseFloat(lonInput.value);
        if (!isFinite(lat) || !isFinite(lon)) {
          msg.textContent = 'Enter latitude and longitude as decimal degrees.';
          msg.classList.add('error');
          return;
        }
        save.disabled = true;
        msg.classList.remove('error');
        msg.textContent = 'Writing EXIF data…';
        try {
          await saveLocation(id, lat, lon);
          msg.textContent = 'Coordinates written to the file.';
          setStatus('Coordinates written to ' + img.filename + '.');
        } catch (err) {
          msg.textContent = err.message;
          msg.classList.add('error');
        } finally {
          save.disabled = false;
        }
      });

      return root;
    }

    function createMarker(img) {
      const marker = L.marker([img.lat, img.lon], { draggable: EDITABLE, autoPan: true });
      marker.bindPopup(function () { return popupContent(img.id); }, {
        maxWidth: 820,
        minWidth: 240,
        offset: [0, -10]
      });
      if (EDITABLE) {
        marker.on('dragstart', function () { state.dragging = img.id; });
        marker.on('dragend', async function () {
          const pos = marker.getLatLng();
          state.dragging = null;
          try {
            await saveLocation(img.id, pos.lat, L.Util.wrapNum(pos.lng, [-180, 180], true));
            setStatus('Moved ' + img.filename + ' to ' + fmt(pos.lat) + ', ' + fmt(pos.lng) + '.');
          } catch (err) {
            setStatus(err.message, true);
            state.version = -1;
            refresh().catch(function () {});
          }
        });
      }
      return marker;
    }

    function apply(images) {
      const seen = new Set();
      const unplaced = [];

      images.forEach(function (img) {
        seen.add(img.id);
        state.images.set(img.id, img);

        if (img.lat === null || img.lon === null) {
          unplaced.push(img);
          const stale = state.markers.get(img.id);
          if (stale) {
            cluster.removeLayer(stale);
            state.markers.delete(img.id);
          }
          return;
        }

        let marker = state.markers.get(img.id);
        if (!marker) {
          marker = createMarker(img);
          state.markers.set(img.id, marker);
          cluster.addLayer(marker);
        } else if (state.dragging !== img.id) {
          const pos = marker.getLatLng();
          if (Math.abs(pos.lat - img.lat) > 1e-9 || Math.abs(pos.lng - img.lon) > 1e-9) {
            cluster.removeLayer(marker);
            marker.setLatLng([img.lat, img.lon]);
            cluster.addLayer(marker);
          }
        }
      });

      state.images.forEach(function (img, id) {
        if (seen.has(id)) return;
        const marker = state.markers.get(id);
        if (marker) cluster.removeLayer(marker);
        state.markers.delete(id);
        state.images.delete(id);
      });

      renderUnplaced(unplaced);
      el.summary.textContent = images.length + ' photo' + (images.length === 1 ? '' : 's') +
        ' · ' + (images.length - unplaced.length) + ' on the map';

      if (!state.didInitialFit) {
        state.didInitialFit = true;
        fitAll();
      }
      if (state.placingId && !state.images.has(state.placingId)) {
        cancelPlacing();
      }
    }

    function fitAll() {
      const points = [];
      state.images.forEach(function (img) {
        if (img.lat !== null && img.lon !== null) points.push([img.lat, img.lon]);
      });
      if (points.length) {
        map.fitBounds(L.latLngBounds(points).pad(0.05));
      } else {
        map.setView([20, 0], 2);
      }
    }

    function renderUnplaced(unplaced) {
      if (!EDITABLE || !unplaced.length) {
        el.unplacedSection.hidden = true;
        el.unplacedList.textContent = '';
        return;
      }
      el.unplacedSection.hidden = false;
      el.unplacedCount.textContent = String(unplaced.length);
      el.unplacedList.textContent = '';

      unplaced.forEach(function (img) {
        const card = document.createElement('button');
        card.type = 'button';
        card.className = 'thumb-card' + (state.placingId === img.id ? ' active' : '');
        card.title = img.filename;

        const picture = document.createElement('img');
        picture.src = '/thumbnail/' + encodeURIComponent(img.id);
        picture.alt = img.filename;
        picture.loading = 'lazy';

        const label = document.createElement('span');
        label.className = 'name';
        label.textContent = img.filename;

        card.appendChild(picture);
        card.appendChild(label);
        card.addEventListener('click', function () {
          if (state.placingId === img.id) {
            cancelPlacing();
          } else {
            startPlacing(img.id);
          }
        });
        el.unplacedList.appendChild(card);
      });
    }

    /* ---------- placing photos without coordinates ---------- */

    function startPlacing(id) {
      const img = state.images.get(id);
      if (!img) return;
      state.placingId = id;
      el.placeText.textContent = 'Click the map to place ' + img.filename;
      el.placeBanner.hidden = false;
      map.getContainer().classList.add('placing');
      Array.prototype.forEach.call(el.unplacedList.children, function (card) {
        card.classList.toggle('active', card.title === img.filename);
      });
    }

    function cancelPlacing() {
      state.placingId = null;
      el.placeBanner.hidden = true;
      map.getContainer().classList.remove('placing');
      Array.prototype.forEach.call(el.unplacedList.children, function (card) {
        card.classList.remove('active');
      });
    }

    if (EDITABLE) {
      el.placeCancel.addEventListener('click', cancelPlacing);
      document.addEventListener('keydown', function (event) {
        if (event.key === 'Escape' && state.placingId) cancelPlacing();
      });

      map.on('click', async function (event) {
        if (!state.placingId) return;
        const id = state.placingId;
        const img = state.images.get(id);
        cancelPlacing();
        try {
          await saveLocation(id, event.latlng.lat, L.Util.wrapNum(event.latlng.lng, [-180, 180], true));
          setStatus('Placed ' + img.filename + ' at ' + fmt(event.latlng.lat) + ', ' + fmt(event.latlng.lng) + '.');
          const marker = state.markers.get(id);
          if (marker) marker.openPopup();
        } catch (err) {
          setStatus(err.message, true);
        }
      });

      /* ---------- upload: file picker and drag & drop ---------- */

      el.fileInput.addEventListener('change', function () {
        if (el.fileInput.files.length) uploadFiles(el.fileInput.files);
        el.fileInput.value = '';
      });

      let dragDepth = 0;
      function hasFiles(event) {
        return Boolean(event.dataTransfer) &&
          Array.prototype.indexOf.call(event.dataTransfer.types || [], 'Files') !== -1;
      }
      window.addEventListener('dragenter', function (event) {
        if (!hasFiles(event)) return;
        event.preventDefault();
        dragDepth += 1;
        el.dropOverlay.classList.add('visible');
      });
      window.addEventListener('dragover', function (event) {
        if (!hasFiles(event)) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = 'copy';
      });
      window.addEventListener('dragleave', function () {
        dragDepth = Math.max(0, dragDepth - 1);
        if (dragDepth === 0) el.dropOverlay.classList.remove('visible');
      });
      window.addEventListener('drop', function (event) {
        if (!hasFiles(event)) return;
        event.preventDefault();
        dragDepth = 0;
        el.dropOverlay.classList.remove('visible');
        uploadFiles(event.dataTransfer.files);
      });
    }

    /* ---------- polling ---------- */

    async function refresh() {
      const data = await fetchImages();
      if (data.version === state.version) return;
      state.version = data.version;
      apply(data.images);
    }

    refresh().catch(function (err) { setStatus(err.message, true); });
    setInterval(function () { refresh().catch(function () {}); }, POLL_MS);

    {% if enable_geolocate %}
    /* ---------- browser geolocation ---------- */
    const locateLayer = L.layerGroup().addTo(map);

    if (el.locateBtn) {
      el.locateBtn.addEventListener('click', function () {
        locateLayer.clearLayers();
        map.locate({ setView: true, maxZoom: 18 });
      });
    }

    map.on('locationfound', function (event) {
      locateLayer.clearLayers();
      L.circle(event.latlng, {
        radius: event.accuracy / 2,
        color: 'yellow',
        fillColor: 'yellow',
        fillOpacity: 0.3
      }).addTo(locateLayer);
      L.circleMarker(event.latlng, {
        radius: 8,
        color: '#000',
        weight: 1,
        fillColor: 'yellow',
        fillOpacity: 1
      }).addTo(locateLayer);
    });

    map.on('locationerror', function (event) {
      setStatus(event.message, true);
    });
    {% endif %}
  })();
  </script>
</body>
</html>
"""


class DuplicateImageError(Exception):
    """Raised when a file with identical content is already known."""

    def __init__(self, existing: str) -> None:
        super().__init__(f"identical to {existing}")
        self.existing = existing


@dataclass
class ImageMetadata:
    """
    Information stored for every photo.

    id          – SHA-256 of the file when it was registered; used as a
                  stable public identifier. It is deliberately *not*
                  recalculated after a coordinate change, so that links
                  and markers stay valid.
    filename    – base name of the file inside the image directory
    lat/lon     – GPS position in decimal degrees, or None if unknown
    thumb_w/h   – pixel dimensions of the thumbnail
    """

    id: str
    filename: str
    lat: Optional[float]
    lon: Optional[float]
    thumb_w: int
    thumb_h: int


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    p = argparse.ArgumentParser(
        description="Display pictures on a map using their GPS EXIF metadata."
    )
    p.add_argument(
        "directory",
        help="Directory with images",
    )
    p.add_argument(
        "-p",
        "--port",
        type=int,
        default=5000,
        help="Port (default 5000)",
    )
    p.add_argument(
        "-H",
        "--host",
        default="0.0.0.0",
        help="Host (default 0.0.0.0)",
    )
    p.add_argument(
        "-t",
        "--title",
        default="Hendrik's Image Map",
        help="Page title (default: Hendrik's Image Map)",
    )
    p.add_argument(
        "-q",
        "--quality",
        type=int,
        default=85,
        metavar="QUALITY",
        help="JPEG quality for full images (0-100, default 85)",
    )
    p.add_argument(
        "-TW",
        "--thumb-width",
        type=int,
        default=600,
        metavar="PX",
        help="Thumbnail width in pixels (default: 600)",
    )
    p.add_argument(
        "-TH",
        "--thumb-height",
        type=int,
        default=400,
        metavar="PX",
        help="Thumbnail height in pixels (default: 400)",
    )
    p.add_argument(
        "-L",
        "--locate",
        action="store_true",
        help="Enable browser geolocation button on the map",
    )
    p.add_argument(
        "-M",
        "--max-upload",
        type=int,
        default=128,
        metavar="MB",
        help="Maximum size of a single upload request in MB (default: 128)",
    )
    p.add_argument(
        "-R",
        "--read-only",
        action="store_true",
        help="Disable uploading and coordinate editing",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="INFO logging",
    )
    p.add_argument(
        "-vv",
        "--debug",
        action="store_true",
        help="DEBUG logging",
    )
    return p.parse_args()


def setup_logging(verbose: bool, debug: bool) -> None:
    """
    Sets up logging based on verbosity level.
    """
    level = logging.DEBUG if debug else logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _convert_to_degrees(
    value: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
) -> float:
    """
    Converts GPS coordinates from EXIF format to decimal degrees.
    """
    d = value[0][0] / value[0][1] if value[0][1] else 0
    m = value[1][0] / value[1][1] if value[1][1] else 0
    s = value[2][0] / value[2][1] if value[2][1] else 0
    return d + m / 60.0 + s / 3600.0


def _convert_to_rational(
    value: float,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Converts decimal degrees into the EXIF degree/minute/second format.
    """
    value = abs(float(value))
    degrees = int(value)
    minutes_float = (value - degrees) * 60
    minutes = int(minutes_float)
    seconds = int(round((minutes_float - minutes) * 60 * 10000))

    if seconds >= 60 * 10000:
        seconds -= 60 * 10000
        minutes += 1
    if minutes >= 60:
        minutes -= 60
        degrees += 1

    return ((degrees, 1), (minutes, 1), (seconds, 10000))


def _extract_gps_info(exif: dict) -> Tuple[float, float]:
    """
    Extracts GPS coordinates from EXIF data.
    """
    gps = exif.get("GPS", {})
    if not gps:
        raise ValueError("No GPS data found in EXIF")

    lat_val = gps.get(piexif.GPSIFD.GPSLatitude)
    lon_val = gps.get(piexif.GPSIFD.GPSLongitude)
    lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef)
    lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef)

    if not (lat_val and lon_val and lat_ref and lon_ref):
        raise ValueError("Incomplete GPS data in EXIF")

    lat = _convert_to_degrees(lat_val)
    lon = _convert_to_degrees(lon_val)
    if lat_ref in [b"S", "S"]:
        lat = -lat
    if lon_ref in [b"W", "W"]:
        lon = -lon
    return lat, lon


def _dump_exif(exif_dict: dict) -> bytes:
    """
    Serializes an EXIF dictionary, dropping only those entries that piexif
    refuses to re-encode. Cameras occasionally write tags with a type piexif
    cannot round-trip; without this the whole update would fail.
    """
    try:
        return piexif.dump(exif_dict)
    except Exception as exc:  # noqa: BLE001
        logging.debug("First EXIF dump attempt failed: %s", exc)

    cleaned = copy.deepcopy(exif_dict)
    for ifd in ("0th", "Exif", "GPS"):
        for tag in list(cleaned.get(ifd, {})):
            probe = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            probe[ifd] = {tag: cleaned[ifd][tag]}
            try:
                piexif.dump(probe)
            except Exception:  # noqa: BLE001
                logging.warning(
                    "Dropping EXIF tag %s of the %s IFD: it cannot be re-encoded",
                    tag,
                    ifd,
                )
                del cleaned[ifd][tag]

    try:
        return piexif.dump(cleaned)
    except Exception as exc:  # noqa: BLE001
        logging.debug("Second EXIF dump attempt failed: %s", exc)

    # Last resort: give up on the embedded EXIF thumbnail.
    cleaned["1st"] = {}
    cleaned["thumbnail"] = None
    logging.warning(
        "Dropping the embedded EXIF thumbnail so the GPS data can be written"
    )
    return piexif.dump(cleaned)


def write_gps_coordinates(path: str, lat: float, lon: float) -> None:
    """
    Rewrites the GPS tags of a JPEG file in place. Only the EXIF segment is
    replaced; the compressed image data and all other metadata are copied
    over byte for byte.
    """
    try:
        exif_dict = piexif.load(path)
    except Exception as exc:  # noqa: BLE001
        logging.debug("No readable EXIF block in '%s' (%s), creating one", path, exc)
        exif_dict = {}

    for key in ("0th", "Exif", "GPS", "1st"):
        if not isinstance(exif_dict.get(key), dict):
            exif_dict[key] = {}
    exif_dict.setdefault("thumbnail", None)

    gps = exif_dict["GPS"]
    gps[piexif.GPSIFD.GPSVersionID] = (2, 0, 0, 0)
    gps[piexif.GPSIFD.GPSLatitudeRef] = "N" if lat >= 0 else "S"
    gps[piexif.GPSIFD.GPSLatitude] = _convert_to_rational(lat)
    gps[piexif.GPSIFD.GPSLongitudeRef] = "E" if lon >= 0 else "W"
    gps[piexif.GPSIFD.GPSLongitude] = _convert_to_rational(lon)

    exif_bytes = _dump_exif(exif_dict)

    # Write into a temporary file first so an error never truncates the original.
    tmp_path = f"{path}.tmp-{os.getpid()}-{threading.get_ident()}"
    try:
        piexif.insert(exif_bytes, path, tmp_path)
        try:
            shutil.copystat(path, tmp_path)
        except OSError as exc:
            logging.debug("Could not copy file stats for '%s': %s", path, exc)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def _sha256_of_file(path: str) -> str:
    """
    Calculates the SHA-256 hash of a file in chunks.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _unique_path(directory: str, filename: str) -> str:
    """
    Returns a path inside directory that does not exist yet, appending a
    counter to the file name if necessary.
    """
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(directory, filename)
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{base}-{counter}{ext}")
        counter += 1
    return candidate


def find_image_files(directory: str) -> List[str]:
    """
    Returns a list of absolute paths to supported images in directory.
    """
    files: List[str] = []

    if not os.path.isdir(directory):
        logging.error("‘%s’ is not a directory", directory)
        return files

    for entry in os.scandir(directory):
        if entry.is_file() and entry.name.lower().endswith(SUPPORTED_EXTENSIONS):
            files.append(entry.path)

    return sorted(files)


class ImageStore:
    """
    Holds every known image together with its pre-rendered JPEGs.

    All mutating operations bump a version counter; the browser polls that
    counter and only redraws when something actually changed.
    """

    def __init__(
        self, directory: str, thumb_size: Tuple[int, int], quality: int
    ) -> None:
        self.directory = os.path.abspath(directory)
        self.thumb_size = thumb_size
        self.quality = quality

        self._lock = threading.RLock()
        self._order: List[str] = []
        self._meta: Dict[str, ImageMetadata] = {}
        self._paths: Dict[str, str] = {}
        self._thumbnails: Dict[str, bytes] = {}
        self._full_images: Dict[str, bytes] = {}
        self._version = 0

    # -------------------- reading --------------------

    def snapshot(self) -> Tuple[int, List[dict]]:
        """
        Returns the current version and the metadata of every image.
        """
        with self._lock:
            return self._version, [asdict(self._meta[i]) for i in self._order]

    def thumbnail(self, image_id: str) -> Optional[bytes]:
        with self._lock:
            return self._thumbnails.get(image_id)

    def full_image(self, image_id: str) -> Optional[bytes]:
        with self._lock:
            return self._full_images.get(image_id)

    # -------------------- writing --------------------

    def load_directory(self) -> None:
        """
        Registers every supported image that is already in the directory.
        """
        for path in find_image_files(self.directory):
            try:
                self.register(path)
            except DuplicateImageError as exc:
                logging.info("Skipping '%s': %s", os.path.basename(path), exc)
            except Exception as exc:  # noqa: BLE001
                logging.warning(
                    "Error processing '%s': %s", os.path.basename(path), exc
                )

    def register(self, path: str) -> ImageMetadata:
        """
        Reads an image from disk, renders its derivatives and adds it to the
        store. Raises DuplicateImageError if the exact file is already known.
        """
        image_id = _sha256_of_file(path)
        with self._lock:
            if image_id in self._meta:
                raise DuplicateImageError(self._meta[image_id].filename)

        lat, lon, thumb_bytes, full_bytes, thumb_size = self._render(path)
        meta = ImageMetadata(
            id=image_id,
            filename=os.path.basename(path),
            lat=lat,
            lon=lon,
            thumb_w=thumb_size[0],
            thumb_h=thumb_size[1],
        )

        with self._lock:
            if image_id in self._meta:
                raise DuplicateImageError(self._meta[image_id].filename)
            self._meta[image_id] = meta
            self._paths[image_id] = path
            self._thumbnails[image_id] = thumb_bytes
            self._full_images[image_id] = full_bytes
            self._order.append(image_id)
            self._version += 1

        logging.info(
            "Registered '%s' (%s)",
            meta.filename,
            "no GPS data" if lat is None else f"{lat:.6f}, {lon:.6f}",
        )
        return replace(meta)

    def add_upload(self, filename: str, data: bytes) -> ImageMetadata:
        """
        Validates an uploaded JPEG, stores it in the image directory and
        registers it. Returns the metadata of the new image.
        """
        safe_name = secure_filename(os.path.basename(filename or "")) or "upload.jpg"
        if not safe_name.lower().endswith(SUPPORTED_EXTENSIONS):
            safe_name = f"{safe_name}.jpg"

        if not data:
            raise ValueError("the file is empty")

        try:
            with Image.open(io.BytesIO(data)) as probe:
                image_format = (probe.format or "").upper()
                probe.verify()
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"not a readable image ({exc})") from exc

        if image_format not in ("JPEG", "MPO"):
            raise ValueError(f"{image_format or 'unknown format'} is not supported")

        digest = hashlib.sha256(data).hexdigest()

        with self._lock:
            if digest in self._meta:
                raise DuplicateImageError(self._meta[digest].filename)
            target = _unique_path(self.directory, safe_name)
            with open(target, "wb") as handle:
                handle.write(data)

        try:
            return self.register(target)
        except Exception:
            try:
                os.remove(target)
            except OSError:
                pass
            raise

    def set_location(self, image_id: str, lat: float, lon: float) -> ImageMetadata:
        """
        Writes new GPS coordinates into the file on disk and updates the
        in-memory metadata. Raises KeyError for an unknown image.
        """
        with self._lock:
            meta = self._meta.get(image_id)
            path = self._paths.get(image_id)
            if meta is None or path is None:
                raise KeyError(image_id)

            write_gps_coordinates(path, lat, lon)

            meta.lat = lat
            meta.lon = lon
            self._version += 1
            logging.info("Updated '%s' to %.6f, %.6f", meta.filename, lat, lon)
            return replace(meta)

    # -------------------- helpers --------------------

    def _render(
        self, path: str
    ) -> Tuple[Optional[float], Optional[float], bytes, bytes, Tuple[int, int]]:
        """
        Extracts the GPS position and renders the full-size JPEG and the
        thumbnail of a single file.
        """
        with Image.open(path) as img:
            lat: Optional[float] = None
            lon: Optional[float] = None

            exif_bytes = img.info.get("exif")
            if exif_bytes:
                try:
                    lat, lon = _extract_gps_info(piexif.load(exif_bytes))
                except Exception as exc:  # noqa: BLE001
                    logging.debug(
                        "No usable GPS data in '%s': %s", os.path.basename(path), exc
                    )

            oriented = ImageOps.exif_transpose(img)

            buf_full = io.BytesIO()
            oriented.convert("RGB").save(
                buf_full, format="JPEG", quality=self.quality, optimize=True
            )

            thumb = oriented.copy()
            thumb.thumbnail(self.thumb_size)
            width, height = thumb.size
            buf_thumb = io.BytesIO()
            thumb.convert("RGB").save(buf_thumb, format="JPEG", quality=85)

        return lat, lon, buf_thumb.getvalue(), buf_full.getvalue(), (width, height)


def create_flask_app(
    store: ImageStore,
    title: str,
    enable_geolocate: bool,
    editable: bool,
    max_upload_bytes: int,
) -> Flask:
    """
    Creates a Flask web application to display and edit images on a map.
    """
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = max_upload_bytes

    def _image_response(data: Optional[bytes]) -> Response:
        if data is None:
            abort(404)
        response = Response(data, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "public, max-age=3600"
        return response

    @app.route("/")
    def index():
        return render_template_string(
            TEMPLATE,
            page_title=title,
            enable_geolocate=enable_geolocate,
            editable=editable,
        )

    @app.get("/api/images")
    def api_images():
        version, images = store.snapshot()
        return jsonify(version=version, images=images)

    @app.get("/image/<path:filehash>")
    def serve_image(filehash: str):
        return _image_response(store.full_image(filehash))

    @app.get("/thumbnail/<path:filehash>")
    def serve_thumbnail(filehash: str):
        """
        Serves a pre-generated thumbnail for the given identifier.
        """
        return _image_response(store.thumbnail(filehash))

    @app.post("/api/upload")
    def api_upload():
        if not editable:
            return jsonify(error="This map is read-only."), 403

        uploads = request.files.getlist("files")
        if not uploads:
            return jsonify(error="No files were received."), 400

        added: List[dict] = []
        skipped: List[dict] = []

        for storage in uploads:
            name = os.path.basename(storage.filename or "unnamed")
            try:
                meta = store.add_upload(name, storage.read())
                added.append(asdict(meta))
            except DuplicateImageError as exc:
                skipped.append(
                    {"name": name, "reason": f"already on the map as {exc.existing}"}
                )
            except Exception as exc:  # noqa: BLE001
                logging.warning("Upload of '%s' failed: %s", name, exc)
                skipped.append({"name": name, "reason": str(exc)})

        version, _ = store.snapshot()
        return jsonify(version=version, added=added, skipped=skipped)

    @app.post("/api/images/<image_id>/location")
    def api_set_location(image_id: str):
        if not editable:
            return jsonify(error="This map is read-only."), 403

        payload = request.get_json(silent=True) or {}
        try:
            lat = float(payload["lat"])
            lon = float(payload["lon"])
        except (KeyError, TypeError, ValueError):
            return jsonify(error="Send lat and lon as decimal numbers."), 400

        if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
            return jsonify(
                error="Latitude must be between -90 and 90, longitude between -180 and 180."
            ), 400

        try:
            meta = store.set_location(image_id, lat, lon)
        except KeyError:
            return jsonify(error="This photo is not on the map any more."), 404
        except Exception as exc:  # noqa: BLE001
            logging.error("Could not write GPS data for %s: %s", image_id, exc)
            return jsonify(error=f"Could not write the EXIF data: {exc}"), 500

        version, _ = store.snapshot()
        return jsonify(version=version, image=asdict(meta))

    @app.errorhandler(413)
    def too_large(_error):
        limit_mb = max_upload_bytes // (1024 * 1024)
        return jsonify(error=f"The upload is larger than {limit_mb} MB."), 413

    return app


def main() -> None:
    """
    Main entry point.
    """
    args = parse_arguments()
    setup_logging(args.verbose, args.debug)

    if not os.path.isdir(args.directory):
        raise SystemExit(f"‘{args.directory}’ is not a directory")

    editable = not args.read_only
    if editable and not os.access(args.directory, os.W_OK):
        logging.warning(
            "‘%s’ is not writable, uploading and editing will fail", args.directory
        )

    store = ImageStore(
        directory=args.directory,
        thumb_size=(args.thumb_width, args.thumb_height),
        quality=args.quality,
    )
    store.load_directory()

    _, images = store.snapshot()
    located = sum(1 for image in images if image["lat"] is not None)
    logging.info(
        "Loaded %d images from ‘%s’, %d of them with GPS data.",
        len(images),
        args.directory,
        located,
    )

    app = create_flask_app(
        store=store,
        title=args.title,
        enable_geolocate=args.locate,
        editable=editable,
        max_upload_bytes=args.max_upload * 1024 * 1024,
    )
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
