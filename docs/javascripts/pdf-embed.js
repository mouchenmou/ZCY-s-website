/**
 * PDF Embed Handler for MkDocs
 * Intercepts PDF links in navigation and embeds them inline within the page
 * instead of navigating directly to the PDF file
 */

(function() {
  'use strict';

  // Wait for DOM to be ready
  document.addEventListener('DOMContentLoaded', function() {
    setupPdfEmbedding();
  });

  function setupPdfEmbedding() {
    // Find all navigation links that point to PDF files
    const navLinks = document.querySelectorAll('.md-nav__link, .md-sidebar__scrollwrap a');

    navLinks.forEach(function(link) {
      const href = link.getAttribute('href');
      if (href && href.toLowerCase().endsWith('.pdf')) {
        link.addEventListener('click', function(e) {
          e.preventDefault();
          e.stopPropagation();
          embedPdfInline(href);
        });
      }
    });

    // Also handle direct PDF URL access
    if (window.location.pathname.toLowerCase().endsWith('.pdf')) {
      const pdfPath = window.location.pathname;
      embedPdfInline(pdfPath, true);
    }
  }

  function embedPdfInline(pdfPath, isDirectAccess) {
    // Get the main content area
    const mainContent = document.querySelector('.md-content');
    if (!mainContent) return;

    // Clear existing content
    const contentInner = mainContent.querySelector('.md-content__inner');
    if (contentInner) {
      contentInner.innerHTML = '';
    }

    // Create PDF viewer container
    const pdfViewer = document.createElement('div');
    pdfViewer.className = 'pdf-embed-container';
    pdfViewer.style.cssText = `
      width: 100%;
      height: calc(100vh - 120px);
      border: none;
      border-radius: 8px;
      overflow: hidden;
      background: #f5f5f5;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    `;

    // Create iframe for PDF
    const iframe = document.createElement('iframe');
    iframe.src = pdfPath;
    iframe.style.cssText = `
      width: 100%;
      height: 100%;
      border: none;
    `;

    pdfViewer.appendChild(iframe);

    // Add to content
    if (contentInner) {
      contentInner.appendChild(pdfViewer);
    } else {
      mainContent.appendChild(pdfViewer);
    }

    // Add custom styles
    addPdfEmbedStyles();

    // Update page title if possible
    const pdfName = pdfPath.split('/').pop().replace('.pdf', '');
    const pageTitle = document.querySelector('.md-typeset h1');
    if (pageTitle) {
      pageTitle.textContent = decodeURIComponent(pdfName);
    }
  }

  function addPdfEmbedStyles() {
    // Add styles only once
    if (document.getElementById('pdf-embed-styles')) return;

    const style = document.createElement('style');
    style.id = 'pdf-embed-styles';
    style.textContent = `
      .pdf-embed-container {
        position: relative;
        margin: 1rem 0;
      }

      .pdf-embed-container::before {
        content: 'PDF Document';
        position: absolute;
        top: -25px;
        left: 0;
        font-size: 0.75rem;
        color: var(--md-default-fg-color--light);
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }

      /* Warm theme specific styles */
      [data-md-color-scheme="warm"] .pdf-embed-container {
        background: #f0e6d2;
        border: 1px solid rgba(0,0,0,0.1);
      }

      [data-md-color-scheme="warm"] .pdf-embed-container iframe {
        background: white;
      }

      /* Dark theme styles */
      [data-md-color-scheme="slate"] .pdf-embed-container {
        background: #1e1e1e;
        border: 1px solid rgba(255,255,255,0.1);
      }

      [data-md-color-scheme="slate"] .pdf-embed-container iframe {
        background: #2d2d2d;
      }

      /* Mobile responsive */
      @media screen and (max-width: 768px) {
        .pdf-embed-container {
          height: calc(100vh - 100px);
          margin: 0.5rem 0;
        }
      }
    `;

    document.head.appendChild(style);
  }
})();
