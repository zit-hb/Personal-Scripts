#!/usr/bin/env python3

# -------------------------------------------------------
# Script: extract_links.py
#
# Description:
# A web application that extracts and visualizes links from URLs.
# It downloads content from URLs, extracts links, and presents them
# in an interactive graph visualization.
#
# Usage:
#   ./extract_links.py [options]
#
# Options:
#   -p, --port PORT              Port to run the server on (default: 5000).
#   -H, --host HOST              Host to run the server on (default: 0.0.0.0).
#   -m, --max-size SIZE          Maximum download size in MB (default: 100).
#   -s, --schemes SCHEME         URL schemes to consider (default: http,https,ftp).
#                                Can be specified multiple times.
#   -i, --include-filetype TYPE  File types to include (if specified, only these types will be processed).
#                                Can be specified multiple times.
#   -e, --exclude-filetype TYPE  File types to exclude. If not set, common image and video files are excluded by default.
#                                Can be specified multiple times.
#   -a, --user-agent AGENT       User agent to use for requests.
#   -v, --verbose                Enable verbose logging (INFO level).
#   -vv, --debug                 Enable debug logging (DEBUG level).
#
# Template: ubuntu24.04
#
# Requirements:
#   - flask (install via: pip install flask==3.1.0)
#   - requests (install via: pip install requests==2.32.3)
#   - beautifulsoup4 (install via: pip install beautifulsoup4==4.13.3)
#   - lxml (install via: pip install lxml==5.3.1)
#   - PyPDF2 (install via: pip install PyPDF2==3.0.1)
#   - python-docx (install via: pip install python-docx==1.1.2)
#
# -------------------------------------------------------
# © 2025 Hendrik Buchwald. All rights reserved.
# -------------------------------------------------------

import argparse
import json
import logging
import os
import re
import tempfile
import urllib.parse
from typing import Dict, List, Optional, Tuple, Any
import gzip
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, Response

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hendriks Link Extractor</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div id="app">
        <div id="landing" class="active">
            <div class="container">
                <h1>Link Extractor</h1>
                <div class="form-container">
                    <form id="url-form">
                        <input type="text" id="url-input" placeholder="Enter a URL (e.g., https://example.com)">
                        <button type="submit">Extract Links</button>
                    </form>
                    <div class="divider">or</div>
                    <div class="drop-area" id="drop-area">
                        <p>Drop a file here or</p>
                        <label for="file-input" class="file-label">Choose file</label>
                        <input type="file" id="file-input" style="display:none;">
                    </div>
                </div>
            </div>
        </div>
        
        <div id="graph-view">
            <div class="toolbar">
                <div class="left">
                    <button id="reset-btn" class="toolbar-btn">Reset</button>
                </div>
                <div class="right">
                    <button id="save-btn" class="toolbar-btn">Save</button>
                </div>
            </div>
            <div id="graph-container"></div>
            <div id="node-info">
                <div class="node-info-content">
                    <h3>Node Information</h3>
                    <p id="node-url"><strong>URL:</strong> <span></span></p>
                    <p id="node-links"><strong>Links:</strong> <span></span></p>
                    <div class="node-actions">
                        <button id="copy-url-btn">Copy URL</button>
                        <button id="expand-node-btn">Expand Node</button>
                        <button id="toggle-domain-btn" style="display:none">Show/Hide Children</button>
                    </div>
                </div>
            </div>
        </div>
        
        <div id="loading-overlay">
            <div class="spinner"></div>
            <p id="loading-message">Processing...</p>
        </div>
        
        <div id="confirmation-modal" class="modal">
            <div class="modal-content">
                <h3>Unsaved Changes</h3>
                <p>You have unsaved changes. Do you want to continue?</p>
                <div class="modal-buttons">
                    <button id="confirm-yes">Yes</button>
                    <button id="confirm-no">No</button>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="script.js"></script>
</body>
</html>"""

STYLES_CSS = """* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

body {
    background-color: #f8f9fa;
    color: #333;
    height: 100vh;
    overflow: hidden;
}

#app {
    height: 100vh;
    display: flex;
    flex-direction: column;
}

/* Landing page */
#landing {
    display: none;
    height: 100%;
    justify-content: center;
    align-items: center;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

#landing.active {
    display: flex;
}

.container {
    max-width: 800px;
    width: 90%;
    background: white;
    border-radius: 12px;
    padding: 40px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
}

h1 {
    text-align: center;
    margin-bottom: 30px;
    color: #2c3e50;
    font-weight: 300;
    font-size: 32px;
}

.form-container {
    display: flex;
    flex-direction: column;
}

#url-form {
    display: flex;
    flex-direction: column;
    gap: 15px;
}

input[type="text"] {
    padding: 12px 15px;
    border: 1px solid #ddd;
    border-radius: 6px;
    font-size: 16px;
    transition: border-color 0.3s;
}

input[type="text"]:focus {
    border-color: #3498db;
    outline: none;
}

button {
    background-color: #3498db;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 12px 15px;
    font-size: 16px;
    cursor: pointer;
    transition: background-color 0.3s;
}

button:hover {
    background-color: #2980b9;
}

.divider {
    margin: 25px 0;
    text-align: center;
    position: relative;
    color: #888;
}

.divider::before,
.divider::after {
    content: "";
    position: absolute;
    top: 50%;
    width: 45%;
    height: 1px;
    background-color: #ddd;
}

.divider::before {
    left: 0;
}

.divider::after {
    right: 0;
}

.drop-area {
    border: 2px dashed #ddd;
    border-radius: 6px;
    padding: 30px;
    text-align: center;
    transition: border-color 0.3s;
    cursor: pointer;
}

.drop-area:hover,
.drop-area.active {
    border-color: #3498db;
}

.drop-area p {
    margin-bottom: 15px;
    color: #666;
}

.file-label {
    display: inline-block;
    background-color: #f1f1f1;
    color: #555;
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.3s;
}

.file-label:hover {
    background-color: #e1e1e1;
}

/* Graph view */
#graph-view {
    display: none;
    flex-direction: column;
    height: 100%;
}

#graph-view.active {
    display: flex;
}

.toolbar {
    background-color: #2c3e50;
    color: white;
    padding: 10px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.toolbar .left, .toolbar .right {
    display: flex;
    align-items: center;
    gap: 10px;
}

.toolbar-btn {
    background-color: transparent;
    border: 1px solid rgba(255, 255, 255, 0.3);
    padding: 8px 15px;
    border-radius: 4px;
    color: white;
    transition: background-color 0.3s;
}

.toolbar-btn:hover {
    background-color: rgba(255, 255, 255, 0.1);
}

#graph-container {
    flex-grow: 1;
    background-color: #f8f9fa;
    position: relative;
    overflow: hidden;
}

/* Node info panel */
#node-info {
    position: fixed;
    bottom: -250px;
    right: 20px;
    width: 350px;
    background-color: white;
    border-radius: 8px 8px 0 0;
    box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
    transition: bottom 0.3s ease-in-out;
    z-index: 10;
}

#node-info.active {
    bottom: 0;
}

.node-info-content {
    padding: 20px;
}

.node-info-content h3 {
    margin-bottom: 15px;
    color: #2c3e50;
    font-weight: 500;
}

.node-info-content p {
    margin-bottom: 10px;
    word-break: break-all;
}

.node-actions {
    display: flex;
    gap: 10px;
    margin-top: 15px;
}

.node-actions button {
    flex: 1;
    padding: 8px;
    font-size: 14px;
}

/* Loading overlay */
#loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.7);
    display: none;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    z-index: 1000;
}

#loading-overlay.active {
    display: flex;
}

.spinner {
    width: 50px;
    height: 50px;
    border: 5px solid rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    border-top-color: #3498db;
    animation: spin 1s ease-in-out infinite;
    margin-bottom: 20px;
}

@keyframes spin {
    to {
        transform: rotate(360deg);
    }
}

#loading-overlay p {
    color: white;
    font-size: 18px;
}

/* Graph nodes */
.node circle {
    stroke: #fff;
    stroke-width: 2px;
}

.node text {
    font-size: 12px;
    fill: #333;
    text-anchor: middle;
    pointer-events: none;
    font-weight: 400;
}

.node text.domain-label {
    font-size: 14px;
    font-weight: bold;
    fill: #333;
    text-anchor: middle;
    pointer-events: none;
}

.node text.url-label {
    font-size: 11px;
    fill: #666;
    text-anchor: middle;
    pointer-events: none;
}

.link {
    stroke: #999;
    stroke-opacity: 0.6;
    stroke-width: 1.5px;
}

/* Modal */
.modal {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    z-index: 2000;
    justify-content: center;
    align-items: center;
}

.modal.active {
    display: flex;
}

.modal-content {
    background-color: white;
    padding: 25px;
    border-radius: 8px;
    width: 400px;
    max-width: 90%;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
}

.modal-content h3 {
    margin-bottom: 15px;
    color: #2c3e50;
}

.modal-buttons {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    margin-top: 20px;
}

.modal-buttons button {
    padding: 8px 15px;
    font-size: 14px;
}

#confirm-yes {
    background-color: #e74c3c;
}

#confirm-yes:hover {
    background-color: #c0392b;
}"""

SCRIPT_JS = r"""document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const landingDiv = document.getElementById('landing');
    const graphView = document.getElementById('graph-view');
    const urlForm = document.getElementById('url-form');
    const urlInput = document.getElementById('url-input');
    const fileInput = document.getElementById('file-input');
    const dropArea = document.getElementById('drop-area');
    const resetBtn = document.getElementById('reset-btn');
    const saveBtn = document.getElementById('save-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingMessage = document.getElementById('loading-message');
    const nodeInfo = document.getElementById('node-info');
    const nodeUrlDisplay = document.querySelector('#node-url span');
    const nodeLinksDisplay = document.querySelector('#node-links span');
    const copyUrlBtn = document.getElementById('copy-url-btn');
    const expandNodeBtn = document.getElementById('expand-node-btn');
    const toggleDomainBtn = document.getElementById('toggle-domain-btn');
    const confirmationModal = document.getElementById('confirmation-modal');
    const confirmYesBtn = document.getElementById('confirm-yes');
    const confirmNoBtn = document.getElementById('confirm-no');

    // Graph variables
    let graph = {
        nodes: [],
        links: []
    };
    let currentAction = null;
    let hasUnsavedChanges = false;
    let simulation = null;
    let selectedNode = null;
    let linkElements = null;
    let nodeElements = null;
    let rootDomain = "";
    
    // We will use two color scales: pastel for subdomains of root domain, bright for external domains
    const pastelScale = d3.scaleOrdinal(d3.schemePastel2);
    const brightScale = d3.scaleOrdinal(d3.schemeSet2);

    // Maps to store domain => color index
    const internalDomainIndexMap = {};
    const externalDomainIndexMap = {};
    let internalDomainIndex = 0;
    let externalDomainIndex = 0;

    // Determine a minimal 'root domain' from the given host
    function getBaseDomain(hostname) {
        // This is fairly naive, but works okay for most typical domains
        const parts = hostname.split('.');
        if (parts.length <= 2) {
            return hostname;
        }
        return parts.slice(-2).join('.');
    }

    // Extract actual hostname from a URL string
    function getHostname(urlStr) {
        try {
            const parsed = new URL(urlStr);
            return parsed.hostname;
        } catch {
            return "";
        }
    }

    // Determine if the node's domain ends with the rootDomain
    function isInternalDomain(nodeDomain, rootDom) {
        // If nodeDomain ends with rootDom, we consider it subdomain or same domain
        return nodeDomain === rootDom || nodeDomain.endsWith("." + rootDom);
    }

    // Returns a color for a node
    function getNodeColor(d) {
      if (d.root) {
        return '#e74c3c';
      }
      if (d.type === "domain") {
        // Domain nodes get different colors based on collapsed state
        if (d.collapsed) {
          return '#ff9500'; // bright orange for collapsed domains
        }
        return '#2ecc71'; // original green for expanded domains
      }
      // For URL nodes, use the previous logic.
      const domain = getHostname(d.id);
      if (isInternalDomain(domain, rootDomain)) {
        if (!(domain in internalDomainIndexMap)) {
          internalDomainIndexMap[domain] = internalDomainIndex++;
        }
        return pastelScale(internalDomainIndexMap[domain]);
      } else {
        if (!(domain in externalDomainIndexMap)) {
          externalDomainIndexMap[domain] = externalDomainIndex++;
        }
        return brightScale(externalDomainIndexMap[domain]);
      }
    }

    // Initialize graph visualization
    function initializeGraph() {
        const width = document.getElementById('graph-container').clientWidth;
        const height = document.getElementById('graph-container').clientHeight;

        // Clear any existing SVG
        d3.select('#graph-container svg').remove();

        // Create SVG container
        const svg = d3.select('#graph-container')
            .append('svg')
            .attr('width', width)
            .attr('height', height);
        
        // Create a group for zoom/pan
        const g = svg.append('g');

        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                g.attr('transform', event.transform);
            });
        
        svg.call(zoom);

        // Create arrows for directed links
        svg.append('defs').append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 25)
            .attr('refY', 0)
            .attr('orient', 'auto')
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', '#999');

        // Create groups for elements
        g.append('g').attr('class', 'links');
        g.append('g').attr('class', 'nodes');

        linkElements = g.select('.links').selectAll('line');
        nodeElements = g.select('.nodes').selectAll('.node');

        // Create a static simulation with adjusted forces for better spacing
        simulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id)
                .distance(d => {
                    // Increase distance for links connected to domain nodes
                    if (d.source.type === "domain" || d.target.type === "domain") {
                        return 150;  // Increased from 100
                    }
                    return 100;  // Increased from 70
                })
                .strength(0.2))  // Decreased from 0.3 for more flexibility
            .force('charge', d3.forceManyBody()
                .strength(d => {
                    // Stronger repulsion for domain and root nodes
                    if (d.type === "domain" || d.root) {
                        return -1500;  // Increased from -1000
                    }
                    return -500;  // Increased from -300
                })
                .distanceMax(800))  // Increased from 500
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collide', d3.forceCollide()
                .radius(d => {
                    // Larger collision radius for domain and root nodes
                    if (d.type === "domain" || d.root) {
                        return 80;  // Increased from 60
                    }
                    return 50;  // Increased from 40
                })
                .strength(0.9)  // Increased from 0.8
                .iterations(4))  // Increased from 3
            .force('x', d3.forceX(width / 2).strength(0.03))  // Decreased from 0.05
            .force('y', d3.forceY(height / 2).strength(0.03));  // Decreased from 0.05

        simulation.alphaDecay(0.02);
        updateGraph();
    }

    // Update graph visualization
    function updateGraph() {
      const visibleNodes = getVisibleNodes();
      const visibleLinks = getVisibleLinks(visibleNodes);

      // Update links
      linkElements = d3.select('.links')
        .selectAll('line')
        .data(visibleLinks, d => `${(d.source.id || d.source)}-${(d.target.id || d.target)}`);
      linkElements.exit().remove();
      const linkEnter = linkElements.enter()
        .append('line')
        .attr('class', 'link')
        .attr('stroke-width', d => Math.sqrt(d.value || 1))
        .attr('marker-end', 'url(#arrowhead)');
      linkElements = linkEnter.merge(linkElements);

      // Update nodes
      nodeElements = d3.select('.nodes')
        .selectAll('.node')
        .data(visibleNodes, d => d.id);
      nodeElements.exit().remove();
      const nodeEnter = nodeElements.enter()
        .append('g')
        .attr('class', 'node')
        .call(d3.drag()
          .on('start', dragstarted)
          .on('drag', dragged)
          .on('end', dragended))
        .on('click', nodeClicked);
      nodeEnter.append('circle')
        .attr('r', d => {
          if (d.root) return 12;
          if (d.type === "domain") return 10; // special radius for domain nodes
          return 8;
        })
        .attr('fill', d => getNodeColor(d));
      nodeEnter.append('text')
        .attr('dy', -15)
        .attr('class', d => d.type === "domain" ? 'domain-label' : '')
        .style('font-weight', d => d.type === "domain" ? 'bold' : 'normal')
        .style('font-size', d => d.type === "domain" ? '14px' : '12px')
        .text(d => getLabelText(d));
      nodeEnter.append('text')
        .attr('dy', d => d.type === "domain" ? 5 : 20)
        .attr('class', 'url-label')
        .text(d => d.type === "domain" ? "" : getUrlLabel(d));
      nodeEnter.append('title')
        .text(d => d.id);
      nodeElements = nodeEnter.merge(nodeElements);

      simulation.nodes(visibleNodes);
      simulation.force('link').links(visibleLinks);
      simulation.alpha(1).restart();
      simulation.on('tick', ticked);

      nodeElements.select('circle')
        .attr('fill', d => getNodeColor(d));
    }

    // Manually apply final positions
    function ticked() {
        linkElements
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        nodeElements
            .attr('transform', d => `translate(${d.x},${d.y})`);
    }

    // Drag functions
    function dragstarted(event, d) {
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
        ticked();
    }

    function dragended(event, d) {
        d.fx = null;
        d.fy = null;
        ticked();
    }

    // Label text
    function getLabelText(node) {
      if (node.type === "domain") {
        // For domain nodes, show the domain (remove the "domain:" prefix if present)
        return node.domain;
      }
      try {
        const url = new URL(node.id);
        // For non-domain nodes, don't show the hostname again - it's already in the domain node
        return "";
      } catch (e) {
        const maxLength = 30;
        return node.id.length > maxLength ? node.id.substring(0, maxLength) + '...' : node.id;
      }
    }

    // Label path portion: show path for URL nodes
    function getUrlLabel(node) {
      if (node.type === "domain") {
        // Domain nodes don't need a URL label
        return "";
      }
      try {
        const url = new URL(node.id);
        const path = url.pathname + url.search;
        const maxLength = 25;
        return path.length > maxLength ? path.substring(0, maxLength) + '...' : path;
      } catch (e) {
        return '';
      }
    }

    // Label path portion
    function getUrlLabel(node) {
      if (node.type === "domain") {
        // Domain nodes do not need a URL label.
        return "";
      }
      try {
        const url = new URL(node.id);
        const path = url.pathname;
        const maxLength = 20;
        return path.length > maxLength ? path.substring(0, maxLength) + '...' : path;
      } catch (e) {
        return '';
      }
    }

    // Node click => show info
    function nodeClicked(event, d) {
      event.stopPropagation();
      selectedNode = d;
      showNodeInfo(d);
    }

    function showNodeInfo(node) {
      if (node.type === "domain") {
        // For domain nodes
        nodeUrlDisplay.textContent = node.domain;
        nodeLinksDisplay.textContent = graph.links.filter(link => 
          (link.source.id === node.id || link.source === node.id)
        ).length || 0;
        
        // Hide the expand button and show the toggle button
        expandNodeBtn.style.display = "none";
        toggleDomainBtn.style.display = "block";
        
        // Update toggle button text based on current state
        toggleDomainBtn.textContent = node.collapsed ? "Show Children" : "Hide Children";
      } else {
        // For regular URL nodes
        nodeUrlDisplay.textContent = node.id;
        nodeLinksDisplay.textContent = node.linkCount || 0;
        
        // Show the expand button and hide the toggle button
        expandNodeBtn.style.display = "block";
        toggleDomainBtn.style.display = "none";
      }
      
      nodeInfo.classList.add('active');
    }

    // Expand a node
    async function expandNode(url) {
        if (!url) return;
        showLoading(true, "Expanding node...");
        try {
            const response = await fetch('/api/extract', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ url })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to extract links');
            }

            showLoading(true, "Processing expanded links...");
            const data = await response.json();
            addNodesToGraph(data);
        } catch (error) {
            console.error('Error expanding node:', error);
            alert('Error: ' + error.message);
        } finally {
            showLoading(false);
        }
    }

    function getVisibleNodes() {
      return graph.nodes.filter(function(node) {
        // If the node has a parent (i.e. is a URL child of a domain),
        // check if that parent is collapsed.
        if (node.parent) {
          const parent = graph.nodes.find(n => n.id === node.parent);
          if (parent && parent.collapsed) {
            return false;
          }
        }
        return true;
      });
    }

    function getVisibleLinks(visibleNodes) {
      const visibleIds = new Set(visibleNodes.map(n => n.id));
      return graph.links.filter(link => {
        // The link’s source/target might be objects or ids.
        const src = link.source.id || link.source;
        const tgt = link.target.id || link.target;
        return visibleIds.has(src) && visibleIds.has(tgt);
      });
    }

    // Update the graph with new data (reset + load)
    function updateGraphWithData(data) {
      // Clear domain index maps so new extracts won't conflict
      for (const key in internalDomainIndexMap) delete internalDomainIndexMap[key];
      for (const key in externalDomainIndexMap) delete externalDomainIndexMap[key];
      internalDomainIndex = 0;
      externalDomainIndex = 0;

      // Reset the graph
      graph = { nodes: [], links: [] };

      // Identify root domain from data.url (used for coloring URL nodes)
      try {
        const parsed = new URL(data.url);
        rootDomain = getBaseDomain(parsed.hostname);
      } catch {
        rootDomain = "";
      }

      // Add the root node (the page that was crawled)
      graph.nodes.push({
        id: data.url,
        root: true
      });

      // Create a dictionary for domain nodes
      const domainNodes = {};
      // Create a dictionary to track all URL nodes we've already created
      const urlNodes = {};

      // Process each link in the extracted data.
      if (data.links && Object.keys(data.links).length > 0) {
        for (const [url, metadata] of Object.entries(data.links)) {
          // Determine the domain for the URL
          const domain = getHostname(url);
          if (!domain) continue;

          // If we haven't created a domain node for this domain yet, do so.
          if (!domainNodes[domain]) {
            const domainNode = {
              id: "domain:" + domain,
              type: "domain",
              domain: domain,
              collapsed: false // initially expanded
            };
            domainNodes[domain] = domainNode;
            graph.nodes.push(domainNode);
            
            // Link from the root node to the domain node.
            graph.links.push({
              source: data.url,
              target: domainNode.id,
              value: 1
            });
          }
          
          // Check if we've already created a node for this URL
          if (!urlNodes[url]) {
            // Create the URL node as a child of the domain node.
            const urlNode = {
              id: url,
              count: metadata.count,
              type: "url", // mark this as a normal URL node
              parent: "domain:" + domain
            };
            graph.nodes.push(urlNode);
            urlNodes[url] = urlNode;
            
            // Link from the domain node to the URL node.
            graph.links.push({
              source: "domain:" + domain,
              target: url,
              value: metadata.count
            });
          }
        }
      }
      
      if (simulation) {
        simulation.stop();
      }
      initializeGraph();
      hasUnsavedChanges = true;
    }

    // Add newly fetched nodes/links to existing graph
    function addNodesToGraph(data) {
      // Find the node that was just expanded
      const parentNode = graph.nodes.find(n => n.id === data.url);
      if (parentNode) {
        parentNode.linkCount = Object.keys(data.links || {}).length;
      } else {
        graph.nodes.push({
          id: data.url,
          linkCount: Object.keys(data.links || {}).length
        });
      }

      // Domain nodes dictionary - global across all nodes to avoid duplicates
      const domainNodesGlobal = {};

      if (data.links && Object.keys(data.links).length > 0) {
        for (const [url, metadata] of Object.entries(data.links)) {
          const domain = getHostname(url);
          if (!domain) continue;
          
          // Use the domain as the ID, not a compound ID
          const domainNodeId = "domain:" + domain;
          
          // Check if domain node already exists in the graph
          let domainNode = graph.nodes.find(n => n.id === domainNodeId);
          if (!domainNode) {
            domainNode = {
              id: domainNodeId,
              type: "domain",
              domain: domain,
              collapsed: false
            };
            graph.nodes.push(domainNode);
            domainNodesGlobal[domain] = domainNode;
          }
          
          // Make sure there's a link from the parent to this domain
          const existingLink = graph.links.find(link => 
            (link.source.id === parentNode.id || link.source === parentNode.id) && 
            (link.target.id === domainNodeId || link.target === domainNodeId)
          );
          
          if (!existingLink) {
            graph.links.push({
              source: parentNode.id,
              target: domainNodeId,
              value: 1
            });
          }
          
          // Check if a node with this URL already exists in the graph
          const existingUrlNode = graph.nodes.find(n => n.id === url);
          if (!existingUrlNode) {
            // Add the URL node only if it doesn't already exist
            const urlNode = {
              id: url,
              count: metadata.count,
              type: "url",
              parent: domainNodeId
            };
            graph.nodes.push(urlNode);
            
            // Link from the domain node to this URL node
            graph.links.push({
              source: domainNodeId,
              target: url,
              value: metadata.count
            });
          }
        }
      }
      updateGraph();
      simulation.alpha(0.3).alphaDecay(0.05).restart();
      hasUnsavedChanges = true;
    }

    // Switch to graph view
    function switchToGraphView() {
        landingDiv.classList.remove('active');
        graphView.classList.add('active');
    }

    // Switch to landing
    function switchToLanding() {
        graphView.classList.remove('active');
        landingDiv.classList.add('active');
        nodeInfo.classList.remove('active');
        selectedNode = null;
        hasUnsavedChanges = false;
    }

    // Show/hide loading
    function showLoading(show, message = "Processing...") {
        if (show) {
            loadingMessage.textContent = message;
            loadingOverlay.classList.add('active');
        } else {
            loadingOverlay.classList.remove('active');
        }
    }

    // Save graph as JSON
    function saveGraph() {
        const dataStr = JSON.stringify(graph);
        const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
        const exportFileDefaultName = 'graph-data.json';
        const linkElement = document.createElement('a');
        linkElement.setAttribute('href', dataUri);
        linkElement.setAttribute('download', exportFileDefaultName);
        linkElement.click();
        hasUnsavedChanges = false;
    }

    // Load graph from file
    async function loadGraph(file) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            showLoading(true, "Loading graph data...");
            const response = await fetch('/api/load', {
                method: 'POST',
                body: formData
            });
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to load graph');
            }
            const data = await response.json();
            graph = data;

            // Clear domain index maps if needed
            for (const key in internalDomainIndexMap) delete internalDomainIndexMap[key];
            for (const key in externalDomainIndexMap) delete externalDomainIndexMap[key];
            internalDomainIndex = 0;
            externalDomainIndex = 0;
            rootDomain = "";

            switchToGraphView();
            setTimeout(() => {
                initializeGraph();
                hasUnsavedChanges = false;
            }, 0);
        } catch (error) {
            console.error('Error loading graph:', error);
            alert('Error: ' + error.message);
        } finally {
            showLoading(false);
        }
    }

    // Show confirmation modal
    function showConfirmation(action) {
        if (!hasUnsavedChanges) {
            action();
            return;
        }
        currentAction = action;
        confirmationModal.classList.add('active');
    }

    // Event Listeners
    toggleDomainBtn.addEventListener('click', () => {
      if (selectedNode && selectedNode.type === "domain") {
        // Toggle collapsed state
        selectedNode.collapsed = !selectedNode.collapsed;
        
        // Update the button text
        toggleDomainBtn.textContent = selectedNode.collapsed ? "Show Children" : "Hide Children";
        
        // Update the graph to reflect the new state
        updateGraph();
        
        // Apply a gentler alpha and decay it faster
        simulation.alpha(0.3).alphaDecay(0.05).restart();
      }
    });

    urlForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const url = urlInput.value.trim();
        if (url) {
            processUrl(url);
        }
    });

    resetBtn.addEventListener('click', () => {
        showConfirmation(() => {
            switchToLanding();
        });
    });

    saveBtn.addEventListener('click', saveGraph);

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            if (file.name.toLowerCase().endsWith('.json')) {
                showConfirmation(() => {
                    loadGraph(file);
                });
            } else {
                processFile(file);
            }
        }
    });

    dropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropArea.classList.add('active');
    });

    dropArea.addEventListener('dragleave', () => {
        dropArea.classList.remove('active');
    });

    dropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dropArea.classList.remove('active');
        const file = e.dataTransfer.files[0];
        if (file) {
            if (file.name.toLowerCase().endsWith('.json')) {
                showConfirmation(() => {
                    loadGraph(file);
                });
            } else {
                processFile(file);
            }
        }
    });

    dropArea.addEventListener('click', () => {
        fileInput.click();
    });

    copyUrlBtn.addEventListener('click', () => {
        if (selectedNode) {
            navigator.clipboard.writeText(selectedNode.id)
                .then(() => {
                    copyUrlBtn.textContent = 'Copied!';
                    setTimeout(() => {
                        copyUrlBtn.textContent = 'Copy URL';
                    }, 2000);
                })
                .catch(err => {
                    console.error('Could not copy text: ', err);
                });
        }
    });

    expandNodeBtn.addEventListener('click', () => {
        if (selectedNode) {
            expandNode(selectedNode.id);
        }
    });

    confirmYesBtn.addEventListener('click', () => {
        confirmationModal.classList.remove('active');
        if (currentAction) {
            currentAction();
            currentAction = null;
        }
    });

    confirmNoBtn.addEventListener('click', () => {
        confirmationModal.classList.remove('active');
        currentAction = null;
    });

    // Process a URL by fetching
    async function processUrl(url) {
        showLoading(true, "Downloading content...");
        try {
            // Add https:// prefix if no protocol is specified
            if (!url.match(/^[a-z]+:\/\/\S+/i)) {
                url = 'https://' + url;
            }
            
            const response = await fetch('/api/extract', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ url })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to extract links');
            }

            showLoading(true, "Processing extracted links...");
            const data = await response.json();

            // Switch first so container has correct size
            switchToGraphView();

            setTimeout(() => {
                updateGraphWithData(data);
            }, 0);
        } catch (error) {
            console.error('Error processing URL:', error);
            alert('Error: ' + error.message);
        } finally {
            showLoading(false);
        }
    }

    // Process a file by extracting
    async function processFile(file) {
        showLoading(true, "Processing file...");
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/api/extract-file', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to process file');
            }

            showLoading(true, "Processing extracted links...");
            const data = await response.json();

            // Switch first so container has correct size
            switchToGraphView();

            setTimeout(() => {
                updateGraphWithData(data);
            }, 0);
        } catch (error) {
            console.error('Error processing file:', error);
            alert('Error: ' + error.message);
        } finally {
            showLoading(false);
        }
    }

    // Window beforeunload
    window.addEventListener('beforeunload', (e) => {
        if (hasUnsavedChanges) {
            e.preventDefault();
            e.returnValue = '';
            return '';
        }
    });
});"""


@dataclass
class ExtractConfig:
    """Configuration for URL extraction."""

    max_size_mb: int
    schemes: List[str]
    include_filetypes: List[str]
    exclude_filetypes: List[str]
    user_agent: str


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Web application that extracts and visualizes links from URLs."
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=5000,
        help="Port to run the server on (default: 5000).",
    )
    parser.add_argument(
        "-H",
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the server on (default: 0.0.0.0).",
    )
    parser.add_argument(
        "-m",
        "--max-size",
        type=int,
        default=100,
        help="Maximum download size in MB (default: 100).",
    )
    parser.add_argument(
        "-s",
        "--schemes",
        type=str,
        action="append",
        default=["http", "https", "ftp"],
        help="URL schemes to consider (default: http,https,ftp).",
    )
    parser.add_argument(
        "-i",
        "--include-filetype",
        type=str,
        action="append",
        default=[],
        help="File types to include (if specified, only these types will be processed).",
    )
    parser.add_argument(
        "-e",
        "--exclude-filetype",
        type=str,
        action="append",
        default=[],
        help="File types to exclude. If not set, common image and video files are excluded by default.",
    )
    parser.add_argument(
        "-a",
        "--user-agent",
        type=str,
        default=DEFAULT_USER_AGENT,
        help="User agent to use for requests.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (INFO level).",
    )
    parser.add_argument(
        "-vv",
        "--debug",
        action="store_true",
        help="Enable debug logging (DEBUG level).",
    )
    return parser.parse_args()


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """
    Sets up the logging configuration.
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def decompress_gz(file_path: str) -> str:
    """
    Decompresses a .gz file to a temporary file with the underlying file extension.
    """
    base_name = os.path.basename(file_path[:-3])
    _, ext = os.path.splitext(base_name)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    with gzip.open(file_path, "rb") as gz_file:
        data = gz_file.read()
    with open(temp_file.name, "wb") as out_file:
        out_file.write(data)
    return temp_file.name


def determine_file_format(file_path: str) -> str:
    """
    Determines the file format based on file extension.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext == ".txt":
        return "txt"
    elif ext == ".pdf":
        return "pdf"
    elif ext == ".docx":
        return "docx"
    elif ext in [".html", ".htm"]:
        return "html"
    elif ext == ".xml":
        return "xml"
    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"]:
        return "image"
    else:
        return "unknown"


def extract_text_from_txt(file_path: str) -> str:
    """
    Extracts text from a plain text file.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        logging.warning(f"Error reading text file {file_path}: {e}")
        return ""


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file using PyPDF2.
    """
    try:
        from PyPDF2 import PdfReader

        text = ""
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        logging.warning(f"Error extracting text from PDF {file_path}: {e}")
        return ""


def extract_text_from_docx(file_path: str) -> str:
    """
    Extracts text from a Word (.docx) document.
    """
    try:
        import docx

        document = docx.Document(file_path)
        paragraphs = [para.text for para in document.paragraphs]
        return "\n".join(paragraphs)
    except Exception as e:
        logging.warning(f"Error extracting text from DOCX {file_path}: {e}")
        return ""


def extract_text_from_html(file_path: str) -> Tuple[str, List[str]]:
    """
    Extracts text and links from an HTML file using BeautifulSoup.
    Returns a tuple of (text_content, links).
    """
    links = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        soup = BeautifulSoup(content, "lxml")
        text = soup.get_text()

        # Extract links from HTML tags
        for a_tag in soup.find_all("a", href=True):
            links.append(a_tag["href"])
        for link_tag in soup.find_all("link", href=True):
            links.append(link_tag["href"])
        for script_tag in soup.find_all("script", src=True):
            links.append(script_tag["src"])
        for img_tag in soup.find_all("img", src=True):
            links.append(img_tag["src"])
        for iframe_tag in soup.find_all("iframe", src=True):
            links.append(iframe_tag["src"])

        return text, links
    except Exception as e:
        logging.warning(f"Error extracting text from HTML {file_path}: {e}")
        # Try reading as binary if text mode fails
        try:
            with open(file_path, "rb") as f:
                content = f.read()

            soup = BeautifulSoup(content, "lxml", features="html.parser")
            text = soup.get_text()

            for a_tag in soup.find_all("a", href=True):
                links.append(a_tag["href"])
            for link_tag in soup.find_all("link", href=True):
                links.append(link_tag["href"])
            for script_tag in soup.find_all("script", src=True):
                links.append(script_tag["src"])
            for img_tag in soup.find_all("img", src=True):
                links.append(img_tag["src"])
            for iframe_tag in soup.find_all("iframe", src=True):
                links.append(iframe_tag["src"])

            return text, links
        except Exception as e2:
            logging.warning(f"Binary fallback also failed: {e2}")
            # Return empty text but still return any links we might have found
            return "", links


def extract_text_from_unknown(file_path: str) -> str:
    """
    Extracts printable text segments from an unknown file type.
    """
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        pattern = re.compile(rb"[ -~]{4,}")
        matches = pattern.findall(data)
        return "\n".join(match.decode("utf-8", errors="replace") for match in matches)
    except Exception as e:
        logging.warning(f"Error extracting text from unknown file {file_path}: {e}")
        return ""


def extract_content(file_path: str) -> Tuple[str, List[str]]:
    """
    Extracts text content and links from the given file based on its type.
    If the file ends with '.gz', it is decompressed and processed.

    Returns a tuple of (text_content, extracted_links).
    """
    extracted_links: List[str] = []

    if file_path.endswith(".gz"):
        temp_file = decompress_gz(file_path)
        try:
            return extract_content(temp_file)
        finally:
            try:
                os.remove(temp_file)
            except Exception as e:
                logging.warning(f"Could not remove temporary file '{temp_file}': {e}")

    file_format = determine_file_format(file_path)
    logging.debug(f"Determined file format: {file_format}")

    if file_format == "txt":
        return extract_text_from_txt(file_path), extracted_links
    elif file_format == "pdf":
        return extract_text_from_pdf(file_path), extracted_links
    elif file_format == "docx":
        return extract_text_from_docx(file_path), extracted_links
    elif file_format == "html":
        return extract_text_from_html(file_path)
    else:
        return extract_text_from_unknown(file_path), extracted_links


def download_url(url: str, config: ExtractConfig) -> Optional[str]:
    """
    Downloads content from a URL to a temporary file.
    Returns the path to the temporary file or None if download failed.
    """
    try:
        headers = {"User-Agent": config.user_agent}

        parsed_url = urllib.parse.urlparse(url)
        if parsed_url.netloc and "]:" in parsed_url.netloc:
            host, port = parsed_url.netloc.rsplit(":", 1)
            if host.startswith("[") and host.endswith("]"):
                pass
            else:
                logging.warning(f"Potentially malformed IPv6 URL: {url}")

        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        # Determine file extension from Content-Type or URL
        extension = ""
        if "html" in content_type:
            extension = ".html"
        elif "text/plain" in content_type:
            extension = ".txt"
        elif "xml" in content_type:
            extension = ".xml"
        elif "pdf" in content_type:
            extension = ".pdf"
        elif (
            "application/msword" in content_type
            or "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            in content_type
        ):
            extension = ".docx"
        else:
            # Try to get extension from URL path
            url_path = parsed_url.path
            if "." in url_path:
                url_extension = os.path.splitext(url_path)[1].lower()
                if url_extension:
                    extension = url_extension

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            downloaded_bytes = 0
            max_bytes = config.max_size_mb * 1024 * 1024

            for chunk in response.iter_content(chunk_size=8192):
                downloaded_bytes += len(chunk)
                if downloaded_bytes > max_bytes:
                    logging.warning(
                        f"Download aborted: size exceeds {config.max_size_mb}MB limit"
                    )
                    os.unlink(temp_file.name)
                    return None
                temp_file.write(chunk)

            return temp_file.name
    except requests.exceptions.InvalidURL as e:
        logging.error(f"Invalid URL {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error downloading URL {url}: {e}")
        return None


def extract_links_from_text(
    text: str,
    html_links: List[str],
    base_url: str,
    config: ExtractConfig,
) -> Dict[str, Dict[str, Any]]:
    """
    Extracts links from text content and normalizes them.
    Also processes extracted HTML links if provided.

    Returns a dictionary of unique URLs with their occurrence count and metadata.
    """

    # A tighter direct URL regex that stops at common delimiters instead of capturing them.
    #  - Captures URLs starting with http://, https://, ftp://, or www.
    #  - Excludes trailing angle brackets, quotes, or whitespace.
    #  - Example matches: "http://example.com", "www.example.com/foo"
    direct_url_pattern = re.compile(
        r'(?:https?|ftp)://[^\s<>"\'()]+|www\.[^\s<>"\'()]+', re.IGNORECASE
    )

    # Href and src match anything between quotes, ignoring extra whitespace.
    href_pattern = re.compile(r'href\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
    src_pattern = re.compile(r'src\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)

    unique_urls: Dict[str, Dict[str, Any]] = {}

    def add_or_increment(target_url: str, source_type: str) -> None:
        """
        Helper to normalize a URL, then add or increment it in the unique_urls dictionary.
        """
        normalized_url = normalize_url(target_url, base_url, config)
        if not normalized_url:
            return

        if normalized_url in unique_urls:
            unique_urls[normalized_url]["count"] += 1
        else:
            unique_urls[normalized_url] = {
                "count": 1,
                "source": base_url,
                "type": source_type,
            }

    # 1. Direct URLs from the text
    for match in direct_url_pattern.finditer(text):
        url = match.group(0)
        add_or_increment(url, "direct")

    # 2. Href attributes
    for match in href_pattern.finditer(text):
        url = match.group(1)
        add_or_increment(url, "href")

    # 3. Src attributes
    for match in src_pattern.finditer(text):
        url = match.group(1)
        add_or_increment(url, "src")

    # 4. HTML links from an external parser
    for url in html_links:
        add_or_increment(url, "html")

    return unique_urls


def normalize_url(url: str, base_url: str, config: ExtractConfig) -> Optional[str]:
    """
    Normalizes a URL by resolving relative URLs against the base URL.
    Checks if the URL scheme is in the allowed list and if the file type is allowed.
    Returns the normalized URL or None if excluded.
    """
    if url.startswith(("data:", "javascript:", "mailto:", "tel:")):
        return None

    if url.startswith("www."):
        url = "http://" + url

    if not url.startswith(("http://", "https://", "ftp://")):
        try:
            url = urllib.parse.urljoin(base_url, url)
        except Exception as e:
            logging.warning(f"Error joining URL {url} with base {base_url}: {e}")
            return None

    try:
        parsed_url = urllib.parse.urlparse(url)
    except Exception as e:
        logging.warning(f"Error parsing URL {url}: {e}")
        return None

    if parsed_url.scheme and parsed_url.scheme not in config.schemes:
        return None

    if not parsed_url.netloc:
        return None

    path = parsed_url.path.lower()

    # Check if path ends with an excluded extension.
    if any(path.endswith(ext.lower()) for ext in config.exclude_filetypes):
        return None

    # If include_filetypes is set, check if it ends with one of those extensions.
    if config.include_filetypes and not any(
        path.endswith(ext.lower()) for ext in config.include_filetypes
    ):
        return None

    try:
        normalized_url = urllib.parse.urlunparse(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                parsed_url.params,
                parsed_url.query,
                "",  # Remove fragment
            )
        )
        return normalized_url
    except Exception as e:
        logging.warning(f"Error normalizing URL {url}: {e}")
        return None


def setup_app(config: ExtractConfig) -> Flask:
    """
    Sets up the Flask app with routes.
    """
    app = Flask(__name__)

    @app.route("/")
    def index() -> Response:
        return Response(INDEX_HTML, mimetype="text/html")

    @app.route("/styles.css")
    def styles() -> Response:
        return Response(STYLES_CSS, mimetype="text/css")

    @app.route("/script.js")
    def script() -> Response:
        return Response(SCRIPT_JS, mimetype="application/javascript")

    @app.route("/api/extract", methods=["POST"])
    def extract() -> Response:
        data = request.json
        url = data.get("url")

        if not url:
            return jsonify({"error": "No URL provided"}), 400

        logging.info(f"Processing URL: {url}")

        if not url.startswith(("http://", "https://", "ftp://")):
            url = "https://" + url

        temp_file_path = download_url(url, config)
        if not temp_file_path:
            return jsonify({"error": "Failed to download URL"}), 400

        try:
            text_content, html_links = extract_content(temp_file_path)
            links = extract_links_from_text(text_content, html_links, url, config)
            return jsonify({"url": url, "links": links})
        except Exception as e:
            logging.error(f"Error processing URL {url}: {e}")
            return jsonify({"error": f"Error processing URL: {str(e)}"}), 500
        finally:
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logging.warning(
                    f"Could not remove temporary file '{temp_file_path}': {e}"
                )

    @app.route("/api/extract-file", methods=["POST"])
    def extract_from_file():
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            file.save(temp_file.name)
            temp_file.close()

            try:
                text_content, html_links = extract_content(temp_file.name)
                links = extract_links_from_text(
                    text_content, html_links, "file://" + file.filename, config
                )
                return jsonify({"url": file.filename, "links": links})
            finally:
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    logging.warning(
                        f"Could not remove temporary file '{temp_file.name}': {e}"
                    )
        except Exception as e:
            logging.error(f"Error processing uploaded file: {e}")
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500

    @app.route("/api/load", methods=["POST"])
    def load_graph() -> Response:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        try:
            file_content = file.read()
            try:
                graph_data = json.loads(file_content.decode("utf-8"))
                return jsonify(graph_data)
            except UnicodeDecodeError:
                encodings = ["latin-1", "cp1252", "iso-8859-1"]
                for encoding in encodings:
                    try:
                        graph_data = json.loads(file_content.decode(encoding))
                        return jsonify(graph_data)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                return jsonify(
                    {
                        "error": "Unable to decode file contents. Invalid JSON or encoding."
                    }
                ), 400
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON format"}), 400
        except Exception as e:
            logging.error(f"Error loading graph data: {e}")
            return jsonify({"error": f"Error loading graph data: {str(e)}"}), 500

    return app


def main() -> None:
    """
    Main function to orchestrate the link extraction application.
    """
    args = parse_arguments()
    setup_logging(verbose=args.verbose, debug=args.debug)

    default_exclusions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".bmp",
        ".tiff",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".mov",
        ".avi",
        ".webm",
        ".mkv",
    ]

    exclude_filetypes = args.exclude_filetype
    if not exclude_filetypes:
        exclude_filetypes = default_exclusions

    config = ExtractConfig(
        max_size_mb=args.max_size,
        schemes=args.schemes,
        include_filetypes=args.include_filetype,
        exclude_filetypes=exclude_filetypes,
        user_agent=args.user_agent,
    )

    app = setup_app(config)
    logging.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
