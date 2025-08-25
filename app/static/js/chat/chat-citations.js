// chat-citations.js

import { showToast } from "./chat-toast.js";
import { showLoadingIndicator, hideLoadingIndicator } from "./chat-loading-indicator.js";
import { toBoolean } from "./chat-utils.js";
import { fetchFileContent } from "./chat-input-actions.js";
// --- NEW IMPORT ---
import { getDocumentMetadata } from './chat-documents.js';
// ------------------

const chatboxEl = document.getElementById("chatbox");

export function parseDocIdAndPage(citationId) {
  // ... (keep existing implementation)
  const underscoreIndex = citationId.lastIndexOf("_");
  if (underscoreIndex === -1) {
    return { docId: null, pageNumber: null };
  }
  const docId = citationId.substring(0, underscoreIndex);
  const pageNumber = citationId.substring(underscoreIndex + 1);
  return { docId, pageNumber };
}

export function parseCitations(message) {
  // ... (keep existing implementation)
  const citationRegex = /\(Source:\s*([^,]+),\s*Page(?:s)?:\s*([^)]+)\)\s*((?:\[#.*?\]\s*)+)/gi;

  return message.replace(citationRegex, (whole, filename, pages, bracketSection) => {
    let filenameHtml;
    if (/^https?:\/\/.+/i.test(filename.trim())) {
      filenameHtml = `<a href="${filename.trim()}" target="_blank" rel="noopener noreferrer">${filename.trim()}</a>`;
    } else {
      filenameHtml = filename.trim();
    }

    const bracketMatches = bracketSection.match(/\[#.*?\]/g) || [];
    const pageToRefMap = {};

    bracketMatches.forEach((match) => {
      let inner = match.slice(2, -1).trim();
      const refs = inner.split(/[;,]/);
      refs.forEach((r) => {
        let ref = r.trim();
        if (ref.startsWith('#')) ref = ref.slice(1);
        const parts = ref.split('_');
        const pageNumber = parts.pop();
        // Ensure docId part is also captured if needed, though ref is the full ID here
        // const docIdPart = parts.join('_');
        pageToRefMap[pageNumber] = ref; // ref is the full citationId like 'docid_pagenum'
      });
    });

    function getDocPrefix(ref) {
      const underscoreIndex = ref.lastIndexOf('_');
      return underscoreIndex === -1 ? ref : ref.slice(0, underscoreIndex + 1);
    }

    const pagesTokens = pages.split(/,/).map(tok => tok.trim());
    const linkedTokens = pagesTokens.map(token => {
      const dashParts = token.split(/[–—-]/).map(p => p.trim());

      if (dashParts.length === 2 && dashParts[0] && dashParts[1]) {
        const startNum = parseInt(dashParts[0], 10);
        const endNum   = parseInt(dashParts[1], 10);

        if (!isNaN(startNum) && !isNaN(endNum)) {
          let discoveredPrefix = '';
          if (pageToRefMap[startNum]) {
            discoveredPrefix = getDocPrefix(pageToRefMap[startNum]);
          } else if (pageToRefMap[endNum]) {
            discoveredPrefix = getDocPrefix(pageToRefMap[endNum]);
          }

          const increment = startNum <= endNum ? 1 : -1;
          const pageAnchors = [];
          for (let p = startNum; increment > 0 ? p <= endNum : p >= endNum; p += increment) {
            if (!pageToRefMap[p] && discoveredPrefix) {
              pageToRefMap[p] = discoveredPrefix + p;
            }
            // Use the full citation ID (ref) from the map for the anchor
            pageAnchors.push(buildAnchorIfExists(String(p), pageToRefMap[p]));
          }
          return pageAnchors.join(', ');
        }
      }

      const singleNum = parseInt(token, 10);
      if (!isNaN(singleNum)) {
        const ref = pageToRefMap[singleNum];
        return buildAnchorIfExists(token, ref);
      }
      return token;
    });

    const linkedPagesText = linkedTokens.join(', ');
    return `(Source: ${filenameHtml}, Pages: ${linkedPagesText})`;
  });
}


export function buildAnchorIfExists(pageStr, citationId) {
  // ... (keep existing implementation)
   if (!citationId) {
    return pageStr;
  }
  // Ensure citationId doesn't have a leading # if passed accidentally
  const cleanCitationId = citationId.startsWith('#') ? citationId.slice(1) : citationId;
  return `<a href="#" class="citation-link" data-citation-id="${cleanCitationId}" target="_blank" rel="noopener noreferrer">${pageStr}</a>`;
}

// --- MODIFIED: fetchCitedText handles errors more gracefully ---
export function fetchCitedText(citationId) {
  showLoadingIndicator();
  fetch("/api/get_citation", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ citation_id: citationId }),
  })
    .then((response) => {
        if (!response.ok) {
            // Try to parse error message from JSON response if possible
            return response.json().then(errData => {
                // Throw an error that includes the server's message
                throw new Error(errData.error || `Server responded with status ${response.status}`);
            }).catch(() => {
                 // If parsing JSON fails, throw a generic error
                 throw new Error(`Server responded with status ${response.status}`);
            });
        }
        return response.json();
    })
    .then((data) => {
      hideLoadingIndicator();

      // Check for expected data fields explicitly
      if (data.cited_text !== undefined && data.file_name && data.page_number !== undefined) {
        showCitedTextPopup(data.cited_text, data.file_name, data.page_number);
      } else if (data.error) { // Handle explicit errors from server even on 200 OK
         showToast(`Could not retrieve citation: ${data.error}`, "warning");
      } else {
         // Handle cases where the response is OK but data is missing
         console.warn("Received citation response but required data is missing:", data);
         showToast("Citation data incomplete.", "warning");
      }
    })
    .catch((error) => {
      hideLoadingIndicator();
      console.error("Error fetching cited text:", error);
      // Show the error message from the caught error
      showToast(`Error fetching citation: ${error.message}`, "danger");
    });
}

export function showCitedTextPopup(citedText, fileName, pageNumber) {
  // ... (keep existing implementation)
  let modalContainer = document.getElementById("citation-modal");
  if (!modalContainer) {
    modalContainer = document.createElement("div");
    modalContainer.id = "citation-modal";
    modalContainer.classList.add("modal", "fade");
    modalContainer.tabIndex = -1;
    modalContainer.setAttribute("aria-hidden", "true");

    modalContainer.innerHTML = `
      <div class="modal-dialog modal-dialog-scrollable modal-xl modal-fullscreen-sm-down">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title">Source: ${fileName}, Page: ${pageNumber}</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body">
            <pre id="cited-text-content"></pre>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(modalContainer);
  } else {
    const modalTitle = modalContainer.querySelector(".modal-title");
    if (modalTitle) {
      modalTitle.textContent = `Source: ${fileName}, Page: ${pageNumber}`;
    }
  }

  const citedTextContent = document.getElementById("cited-text-content");
  if (citedTextContent) {
    citedTextContent.textContent = citedText;
  }

  const modal = new bootstrap.Modal(modalContainer);
  modal.show();
}

export function showImagePopup(imageSrc) {
  // ... (keep existing implementation)
  let modalContainer = document.getElementById("image-modal");
  if (!modalContainer) {
    modalContainer = document.createElement("div");
    modalContainer.id = "image-modal";
    modalContainer.classList.add("modal", "fade");
    modalContainer.tabIndex = -1;
    modalContainer.setAttribute("aria-hidden", "true");

    modalContainer.innerHTML = `
      <div class="modal-dialog modal-dialog-centered">
        <div class="modal-content">
          <div class="modal-body text-center">
            <img
              id="image-modal-img"
              src=""
              alt="Generated Image"
              class="img-fluid"
            />
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(modalContainer);
  }
  const modalImage = modalContainer.querySelector("#image-modal-img");
  if (modalImage) {
    modalImage.src = imageSrc;
  }
  const modal = new bootstrap.Modal(modalContainer);
  modal.show();
}

// --- MODIFIED: Added citationId parameter and fallback in catch ---
export function showPdfModal(docId, pageNumber, citationId) {
  const fetchUrl = `/view_pdf?doc_id=${encodeURIComponent(docId)}&page=${encodeURIComponent(pageNumber)}`;

  let pdfModal = document.getElementById("pdf-modal");
  if (!pdfModal) {
    pdfModal = document.createElement("div");
    pdfModal.id = "pdf-modal";
    pdfModal.classList.add("modal", "fade");
    pdfModal.tabIndex = -1;
    pdfModal.innerHTML = `
      <div class="modal-dialog modal-dialog-scrollable modal-xl modal-fullscreen-sm-down">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title">Citation +/- one page</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body" style="height:80vh;">
            <iframe
              id="pdf-iframe"
              src=""
              style="width:100%; height:100%; border:none;"
            ></iframe>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(pdfModal);
  }

  showLoadingIndicator();

  fetch(fetchUrl)
    .then(async (resp) => {
      // Keep existing success logic
      if (!resp.ok) {
         // Throw an error to be caught by the .catch block
         const errorText = await resp.text(); // Try to get more info
         throw new Error(`Failed to load PDF. Status: ${resp.status}. ${errorText.substring(0, 100)}`);
      }
       hideLoadingIndicator(); // Hide indicator ONLY on successful fetch response start

      const newPage = resp.headers.get("X-Sub-PDF-Page") || "1";
      const blob = await resp.blob();
      const pdfBlobUrl = URL.createObjectURL(blob);
      const iframeSrc = pdfBlobUrl + `#page=${newPage}`;
      const iframe = pdfModal.querySelector("#pdf-iframe");
      if (iframe) {
        iframe.src = iframeSrc;
        // Ensure modal is shown AFTER iframe src is set
         const modalInstance = new bootstrap.Modal(pdfModal);
         modalInstance.show();
      } else {
          // Should not happen if modal structure is correct
          console.error("PDF iframe element not found after creating modal.");
          showToast("Error displaying PDF viewer.", "danger");
           // Fallback if iframe fails to load? Maybe too complex.
           // fetchCitedText(citationId);
      }

    })
    .catch((error) => {
      // --- FALLBACK LOGIC ---
      hideLoadingIndicator(); // Ensure indicator is hidden on error
      console.error("Error fetching PDF, falling back to text citation:", error);
      // showToast(`Could not load PDF preview: ${error.message}. Falling back to text citation.`, "warning");
      // Call the text-based citation fetcher
      fetchCitedText(citationId);
      // --- END FALLBACK ---

      // Ensure modal doesn't linger if PDF fetch failed before showing
      const maybeModalInstance = bootstrap.Modal.getInstance(pdfModal);
      if (maybeModalInstance) {
          maybeModalInstance.hide();
      }
    });
}
// --------------------------------------------------------------------

// --- MODIFIED: Event Listener Logic ---
if (chatboxEl) {
  chatboxEl.addEventListener("click", (event) => {
    const target = event.target.closest('a'); // Find the nearest ancestor anchor tag

    // Check if it's an inline citation link OR a hybrid citation button
    if (target && (target.matches("a.citation-link") || target.matches("a.citation-button.hybrid-citation-link"))) {
      event.preventDefault();
      const citationId = target.getAttribute("data-citation-id");
      if (!citationId) {
          console.warn("Citation link/button clicked but data-citation-id is missing.");
          showToast("Cannot process citation: Missing ID.", "warning");
          return;
      }

      const { docId, pageNumber } = parseDocIdAndPage(citationId);

      // Safety check: Ensure docId and pageNumber were parsed correctly
      if (!docId || !pageNumber) {
          console.warn(`Could not parse docId/pageNumber from citationId: ${citationId}. Falling back to text citation.`);
          // showToast("Could not identify document source, showing text.", "info");
          fetchCitedText(citationId); // Fallback to text if parsing fails
          return;
      }

      // --- Logic to decide between PDF and Text ---
      const useEnhancedGlobally = toBoolean(window.enableEnhancedCitations);
      let attemptEnhanced = false; // Default to not attempting enhanced

      if (useEnhancedGlobally) {
          // console.log(`Checking metadata for docId: ${docId}`);
          const docMetadata = getDocumentMetadata(docId); // Fetch metadata

          // Decide based on metadata:
          // Attempt enhanced if:
          // 1. Metadata found AND enhanced_citations is NOT explicitly false
          // 2. Metadata not found (assume enhanced might be possible, rely on error fallback)
          if (!docMetadata) {
              // console.log(`Metadata not found for ${docId}, attempting enhanced citation (will fallback on error).`);
              attemptEnhanced = true;
          } else if (docMetadata.enhanced_citations === false) {
              // console.log(`Metadata found for ${docId}, enhanced_citations is false. Using text citation.`);
              attemptEnhanced = false; // Explicitly disabled for this doc
          } else {
              // console.log(`Metadata found for ${docId}, enhanced_citations is true or undefined. Attempting enhanced citation.`);
              attemptEnhanced = true; // Includes cases where metadata exists but enhanced_citations is true, null, or undefined
          }
      } else {
        // console.log("Global enhanced citations disabled. Using text citation.");
        attemptEnhanced = false; // Globally disabled
      }

      // --- Execute based on the decision ---
      if (attemptEnhanced) {
          // console.log(`Attempting PDF Modal for ${docId}, page ${pageNumber}, citationId ${citationId}`);
          // Pass citationId for potential fallback within showPdfModal
          showPdfModal(docId, pageNumber, citationId);
      } else {
          // console.log(`Fetching Text Citation for ${citationId}`);
          // Use text citation if globally disabled OR explicitly disabled for this doc OR if parsing failed earlier
          fetchCitedText(citationId);
      }
      // --- End Logic ---

    } else if (target && target.matches("a.file-link")) { // Keep existing file link logic
      event.preventDefault();
      const fileId = target.getAttribute("data-file-id");
      const conversationId = target.getAttribute("data-conversation-id");
      if (fileId && conversationId) { // Add checks
        fetchFileContent(conversationId, fileId);
      } else {
        console.warn("File link clicked but missing data-file-id or data-conversation-id");
        showToast("Could not open file: Missing information.", "warning");
      }
    } else if (event.target && event.target.classList.contains("generated-image")) { // Keep existing image logic
        // Use event.target directly here as it's the image itself
      const imageSrc = event.target.getAttribute("data-image-src");
      if (imageSrc) {
          showImagePopup(imageSrc);
      }
    }
    // Clicks on web citation buttons (a.citation-button.web-citation-link) are handled
    // natively by the browser because they have a valid href and target="_blank".
    // No specific JS handling needed here unless you want to add tracking etc.
  });
}
// ---------------------------------------