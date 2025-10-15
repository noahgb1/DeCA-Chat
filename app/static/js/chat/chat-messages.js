// chat-messages.js

import { parseCitations } from "./chat-citations.js";
import { renderFeedbackIcons } from "./chat-feedback.js";
import {
  showLoadingIndicatorInChatbox,
  hideLoadingIndicatorInChatbox,
} from "./chat-loading-indicator.js";
import { docScopeSelect, getDocumentMetadata } from "./chat-documents.js";
import { promptSelect } from "./chat-prompts.js";
import {
  createNewConversation,
  selectConversation,
  addConversationToList,
  loadConversations,             // <-- ADDED
} from "./chat-conversations.js";
import { escapeHtml } from "./chat-utils.js";
import { showToast } from "./chat-toast.js";
import { saveUserSetting } from "./chat-layout.js";

export const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const promptSelectionContainer = document.getElementById(
  "prompt-selection-container"
);
const chatbox = document.getElementById("chatbox");
const modelSelect = document.getElementById("model-select");

/* === NEW: streaming/interrupt state === */
let currentChatAbortController = null;
let isRequestInFlight = false;

function setInFlightState(inFlight) {
  isRequestInFlight = inFlight;
  if (userInput) userInput.disabled = inFlight;

  // Morph the single button between Send and Stop
  if (sendBtn) {
    const icon  = sendBtn.querySelector("[data-icon]")  || sendBtn.querySelector("i");
    const label = sendBtn.querySelector("[data-label]") || sendBtn.querySelector("span");
    if (inFlight) {
      sendBtn.classList.remove("btn-primary");
      sendBtn.classList.add("btn-outline-danger");
      sendBtn.title = "Stop generating";
      if (icon) icon.className = "bi bi-stop-fill";
      if (label) label.textContent = "Stop";
    } else {
      sendBtn.classList.add("btn-primary");
      sendBtn.classList.remove("btn-outline-danger");
      sendBtn.title = "Send message";
      if (icon) icon.className = "bi bi-send-fill";
      if (label) label.textContent = "Send";
    }
  }
}

// Safely call the classic-script global from a module
function safeScrollToBottom() {
  try {
    if (typeof window.scrollChatToBottom === "function") {
      window.scrollChatToBottom();
    }
  } catch {}
}

function createCitationsHtml(
  hybridCitations = [],
  webCitations = [],
  messageId
) {
  let citationsHtml = "";
  let hasCitations = false;

  if (hybridCitations && hybridCitations.length > 0) {
    hasCitations = true;
    hybridCitations.forEach((cite, index) => {
      const citationId =
        cite.citation_id || `${cite.chunk_id}_${cite.page_number || index}`;
      const displayText = `${escapeHtml(cite.file_name)}, Page ${
        cite.page_number || "N/A"
      }`;
      citationsHtml += `
              <a href="#"
                 class="btn btn-sm citation-button hybrid-citation-link"
                 data-citation-id="${escapeHtml(citationId)}"
                 title="View source: ${displayText}">
                  <i class="bi bi-file-earmark-text me-1"></i>${displayText}
              </a>`;
    });
  }

  if (webCitations && webCitations.length > 0) {
    hasCitations = true;
    webCitations.forEach((cite) => {
      const displayText = cite.title
        ? escapeHtml(cite.title)
        : escapeHtml(cite.url);
      citationsHtml += `
              <a href="${escapeHtml(
                cite.url
              )}" target="_blank" rel="noopener noreferrer"
                 class="btn btn-sm citation-button web-citation-link"
                 title="View web source: ${displayText}">
                  <i class="bi bi-globe me-1"></i>${displayText}
              </a>`;
    });
  }

  if (hasCitations) {
    return `<div class="citations-container" data-message-id="${escapeHtml(
      messageId
    )}">${citationsHtml}</div>`;
  } else {
    return "";
  }
}

export function loadMessages(conversationId) {
  // **Guard**: don't wipe/replace the chatbox while a stream is in-flight.
  if (isRequestInFlight) {
    console.debug("[loadMessages] Skipped reload because a stream is in-flight.");
    return;
  }

  fetch(`/conversation/${conversationId}/messages`)
    .then((response) => response.json())
    .then((data) => {
      const chatbox = document.getElementById("chatbox");
      if (!chatbox) return;

      chatbox.innerHTML = "";
      console.log(`--- Loading messages for ${conversationId} ---`);
      data.messages.forEach((msg) => {
        console.log(
          `[loadMessages Loop] -------- START Message ID: ${msg.id} --------`
        );
        console.log(`[loadMessages Loop] Role: ${msg.role}`);
        if (msg.role === "user") {
          appendMessage("You", msg.content);
        } else if (msg.role === "assistant") {
          console.log(
            `  [loadMessages Loop] Full Assistant msg object:`,
            JSON.stringify(msg)
          );
          console.log(
            `  [loadMessages Loop] Checking keys: msg.id=${msg.id}, msg.augmented=${msg.augmented}, msg.hybrid_citations exists=${
              "hybrid_citations" in msg
            }, msg.web_search_citations exists=${
              "web_search_citations" in msg
            }`
          );
          const senderType =
            msg.role === "user"
              ? "You"
              : msg.role === "assistant"
              ? "AI"
              : msg.role === "file"
              ? "File"
              : msg.role === "image"
              ? "image"
              : msg.role === "safety"
              ? "safety"
              : "System";

          const arg2 = msg.content;
          const arg3 = msg.model_deployment_name;
          const arg4 = msg.id;
          const arg5 = msg.augmented;
          const arg6 = msg.hybrid_citations;
          const arg7 = msg.web_search_citations;
          console.log(
            `  [loadMessages Loop] Calling appendMessage with -> sender: ${senderType}, id: ${arg4}, augmented: ${arg5} (type: ${typeof arg5}), hybrid_len: ${arg6?.length}, web_len: ${arg7?.length}`
          );

          appendMessage(senderType, arg2, arg3, arg4, arg5, arg6, arg7);
          console.log(
            `[loadMessages Loop] -------- END Message ID: ${msg.id} --------`
          );
        } else if (msg.role === "file") {
          appendMessage("File", msg);
        } else if (msg.role === "image") {
          appendMessage("image", msg.content, msg.model_deployment_name);
        } else if (msg.role === "safety") {
          appendMessage("safety", msg.content);
        }
      });
    })
    .catch((error) => {
      console.error("Error loading messages:", error);
      if (chatbox)
        chatbox.innerHTML = `<div class="text-center p-3 text-danger">Error loading messages.</div>`;
    });
}

export function appendMessage(
  sender,
  messageContent,
  modelName = null,
  messageId = null,
  augmented = false,
  hybridCitations = [],
  webCitations = []
) {
  if (!chatbox || sender === "System") return;

  const messageDiv = document.createElement("div");
  messageDiv.classList.add("mb-2", "message");
  messageDiv.setAttribute("data-message-id", messageId || `msg-${Date.now()}`);

  let avatarImg = "";
  let avatarAltText = "";
  let messageClass = "";
  let senderLabel = "";
  let messageContentHtml = "";

  if (sender === "AI") {
    console.log(`--- appendMessage called for AI ---`);
    console.log(`Message ID: ${messageId}`);
    console.log(`Received augmented: ${augmented} (Type: ${typeof augmented})`);
    console.log(
      `Received hybridCitations:`,
      hybridCitations,
      `(Length: ${hybridCitations?.length})`
    );
    console.log(
      `Received webCitations:`,
      webCitations,
      `(Length: ${webCitations?.length})`
    );

    messageClass = "ai-message";
    avatarAltText = "AI Avatar";
    avatarImg = "/static/images/ai-avatar.png";
    senderLabel = modelName
      ? `AI <span style="color: #6c757d; font-size: 0.8em;">(${modelName})</span>`
      : "AI";

    // Parse content
    let cleaned = (messageContent || "").trim().replace(/\n{3,}/g, "\n\n");
    cleaned = cleaned.replace(/(\bhttps?:\/\/\S+)(%5D|\])+/gi, (_, url) => url);
    const withInlineCitations = parseCitations(cleaned);
    const htmlContent = DOMPurify.sanitize(marked.parse(withInlineCitations));
    const mainMessageHtml = `<div class="message-text">${htmlContent}</div>`;

    // Footer content (Copy, Feedback, Citations)
    const feedbackHtml = renderFeedbackIcons(messageId, currentConversationId);
    const hiddenTextId = `copy-md-${messageId || Date.now()}`;
    const copyButtonHtml = `
            <button class="copy-btn me-2" data-hidden-text-id="${hiddenTextId}" title="Copy AI response as Markdown">
                <i class="bi bi-copy"></i>
            </button>
            <textarea id="${hiddenTextId}" style="display:none;">${escapeHtml(
              withInlineCitations
            )}</textarea>
        `;
    const copyAndFeedbackHtml = `<div class="message-actions d-flex align-items-center">${copyButtonHtml}${feedbackHtml}</div>`;

    const citationsButtonsHtml = createCitationsHtml(
      hybridCitations,
      webCitations,
      messageId
    );

    let citationToggleHtml = "";
    let citationContentContainerHtml = "";

    const shouldShowCitations = augmented && citationsButtonsHtml;
    if (shouldShowCitations) {
      const citationsContainerId = `citations-${messageId || Date.now()}`;
      citationToggleHtml = `<div class="citation-toggle-container"><button class="btn btn-sm btn-outline-secondary citation-toggle-btn" title="Show sources" aria-expanded="false" aria-controls="${citationsContainerId}"><i class="bi bi-journal-text"></i></button></div>`;
      citationContentContainerHtml = `<div class="citations-container mt-2 pt-2 border-top" id="${citationsContainerId}" style="display: none;">${citationsButtonsHtml}</div>`;
    }

    const footerContentHtml = `<div class="message-footer d-flex justify-content-between align-items-center">${copyAndFeedbackHtml}${citationToggleHtml}</div>`;

    messageDiv.innerHTML = `
            <div class="message-content">
                <img src="${avatarImg}" alt="${avatarAltText}" class="avatar">
                <div class="message-bubble">
                    <div class="message-sender">${senderLabel}</div>
                    ${mainMessageHtml}
                    ${citationContentContainerHtml}
                    ${footerContentHtml}
                </div>
            </div>`;

    messageDiv.classList.add(messageClass);
    chatbox.appendChild(messageDiv);

    attachCodeBlockCopyButtons(messageDiv.querySelector(".message-text"));

    const copyBtn = messageDiv.querySelector(".copy-btn");
    copyBtn?.addEventListener("click", () => {
      const hiddenTextarea = document.getElementById(
        copyBtn.dataset.hiddenTextId
      );
      if (!hiddenTextarea) return;
      navigator.clipboard
        .writeText(hiddenTextarea.value)
        .then(() => {
          copyBtn.innerHTML = '<i class="bi bi-check-lg text-success"></i>';
          copyBtn.title = "Copied!";
          setTimeout(() => {
            copyBtn.innerHTML = '<i class="bi bi-copy"></i>';
            copyBtn.title = "Copy AI response as Markdown";
          }, 2000);
        })
        .catch((err) => {
          console.error("Error copying text:", err);
          showToast("Failed to copy text.", "warning");
        });
    });

    const toggleBtn = messageDiv.querySelector(".citation-toggle-btn");
    if (toggleBtn) {
      toggleBtn.addEventListener("click", () => {
        const targetId = toggleBtn.getAttribute("aria-controls");
        const citationsContainer = messageDiv.querySelector(`#${targetId}`);
        if (!citationsContainer) return;
        const isExpanded = citationsContainer.style.display !== "none";
        citationsContainer.style.display = isExpanded ? "none" : "block";
        toggleBtn.setAttribute("aria-expanded", !isExpanded);
        toggleBtn.title = isExpanded ? "Show sources" : "Hide sources";
        toggleBtn.innerHTML = isExpanded
          ? '<i class="bi bi-journal-text"></i>'
          : '<i class="bi bi-chevron-up"></i>';
        if (!isExpanded) {
          safeScrollToBottom();
        }
      });
    }

    safeScrollToBottom();
    return messageDiv; // allow streaming updates
  } else {
    // Determine variables based on sender type
    if (sender === "You") {
      messageClass = "user-message";
      senderLabel = "You";
      avatarAltText = "User Avatar";
      avatarImg = "/static/images/user-avatar.png";
      messageContentHtml = DOMPurify.sanitize(
        marked.parse(escapeHtml(messageContent))
      );
    } else if (sender === "File") {
      messageClass = "file-message";
      senderLabel = "File Added";
      avatarImg = "";
      avatarAltText = "";
      const filename = escapeHtml(messageContent.filename);
      const fileId = escapeHtml(messageContent.id);
      messageContentHtml = `<a href="#" class="file-link" data-conversation-id="${currentConversationId}" data-file-id="${fileId}"><i class="bi bi-file-earmark-arrow-up me-1"></i>${filename}</a>`;
    } else if (sender === "image") {
      messageClass = "image-message";
      senderLabel = modelName
        ? `AI <span style="color: #6c757d; font-size: 0.8em;">(${modelName})</span>`
        : "Image";
      avatarImg = "/static/images/ai-avatar.png";
      avatarAltText = "Generated Image";
      messageContentHtml = `<img src="${messageContent}" alt="Generated Image" class="generated-image" style="width: 170px; height: 170px; cursor: pointer;" data-image-src="${messageContent}" onload="scrollChatToBottom()" />`;
    } else if (sender === "safety") {
      messageClass = "safety-message";
      senderLabel = "Content Safety";
      avatarAltText = "Content Safety Avatar";
      avatarImg = "/static/images/alert.png";
      const linkToViolations = `<br><small><a href="/safety_violations" target="_blank" rel="noopener" style="font-size: 0.85em; color: #6c757d;">View My Safety Violations</a></small>`;
      messageContentHtml = DOMPurify.sanitize(
        marked.parse(messageContent + linkToViolations)
      );
    } else if (sender === "Error") {
      messageClass = "error-message";
      senderLabel = "System Error";
      avatarImg = "/static/images/alert.png";
      avatarAltText = "Error Avatar";
      messageContentHtml = `<span class="text-danger">${escapeHtml(
        messageContent
      )}</span>`;
    } else {
      console.warn("Unknown message sender type:", sender);
      messageClass = "unknown-message";
      senderLabel = "System";
      avatarImg = "/static/images/ai-avatar.png";
      avatarAltText = "System Avatar";
      messageContentHtml = escapeHtml(messageContent);
    }

    messageDiv.classList.add(messageClass);

    messageDiv.innerHTML = `
            <div class="message-content ${
              sender === "You" || sender === "File" ? "flex-row-reverse" : ""
            }">
                ${
                  avatarImg
                    ? `<img src="${avatarImg}" alt="${avatarAltText}" class="avatar">`
                    : ""
                }
                <div class="message-bubble">
                    <div class="message-sender">${senderLabel}</div>
                    <div class="message-text">${messageContentHtml}</div>
                </div>
            </div>`;

    chatbox.appendChild(messageDiv);
    safeScrollToBottom();
  }
}

export function sendMessage() {
  if (!userInput) {
    console.error("User input element not found.");
    return;
  }
  let userText = userInput.value.trim();
  let promptText = "";
  let combinedMessage = "";

  if (
    promptSelectionContainer &&
    promptSelectionContainer.style.display !== "none" &&
    promptSelect &&
    promptSelect.selectedIndex > 0
  ) {
    const selectedOpt = promptSelect.options[promptSelect.selectedIndex];
    promptText = selectedOpt?.dataset?.promptContent?.trim() || "";
  }

  if (userText && promptText) {
    combinedMessage = userText + "\n\n" + promptText;
  } else {
    combinedMessage = userText || promptText;
  }
  combinedMessage = combinedMessage.trim();

  if (!combinedMessage) {
    return;
  }

  const dispatch = (window.USE_STREAMING
    ? actuallySendMessageStream
    : actuallySendMessage);

  if (!currentConversationId) {
    createNewConversation(() => {
      dispatch(combinedMessage);
    });
  } else {
    dispatch(combinedMessage);
  }

  userInput.value = "";
  userInput.style.height = "";
  if (promptSelect) {
    promptSelect.selectedIndex = 0;
  }
  userInput.focus();
}

export function actuallySendMessage(finalMessageToSend) {
  appendMessage("You", finalMessageToSend);
  userInput.value = "";
  userInput.style.height = "";
  showLoadingIndicatorInChatbox();

  const modelDeployment = modelSelect?.value;

  let hybridSearchEnabled = false;
  const sdbtn = document.getElementById("search-documents-btn");
  if (sdbtn && sdbtn.classList.contains("active")) {
    hybridSearchEnabled = true;
  }

  let selectedDocumentId = null;
  let classificationsToSend = null;
  const docSel = document.getElementById("document-select");
  const classificationInput = document.getElementById("classification-select");

  if (docSel) {
    const selectedDocOption = docSel.options[docSel.selectedIndex];
    selectedDocumentId =
      selectedDocOption && selectedDocOption.value !== ""
        ? selectedDocOption.value
        : null;
  }

  if (classificationInput) {
    classificationsToSend =
      classificationInput.value === "N/A" ? null : classificationInput.value;
  }

  let bingSearchEnabled = false;
  const wbbtn = document.getElementById("search-web-btn");
  if (wbbtn && wbbtn.classList.contains("active")) {
    bingSearchEnabled = true;
  }

  let imageGenEnabled = false;
  const igbtn = document.getElementById("image-generate-btn");
  if (igbtn && igbtn.classList.contains("active")) {
    imageGenEnabled = true;
  }

  fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: finalMessageToSend,
      conversation_id: currentConversationId,
      hybrid_search: hybridSearchEnabled,
      selected_document_id: selectedDocumentId,
      classifications: classificationsToSend,
      bing_search: bingSearchEnabled,
      image_generation: imageGenEnabled,
      doc_scope: docScopeSelect ? docScopeSelect.value : "all",
      active_group_id: window.activeGroupId,
      model_deployment: modelDeployment,
    }),
  })
    .then((response) => {
      if (!response.ok) {
        return response
          .json()
          .then((errData) => {
            const error = new Error(
              errData.error || `HTTP error! status: ${response.status}`
            );
            error.status = response.status;
            error.data = errData;
            throw error;
          })
          .catch(() => {
            throw new Error(`HTTP error! status: ${response.status}`);
          });
      }
      return response.json();
    })
    .then((data) => {
      hideLoadingIndicatorInChatbox();

      console.log("--- Data received from /api/chat ---");
      console.log("Full data object:", data);
      console.log(
        `data.augmented: ${data.augmented} (Type: ${typeof data.augmented})`
      );
      console.log("data.hybrid_citations:", data.hybrid_citations);
      console.log("data.web_search_citations:", data.web_search_citations);
      console.log(`data.message_id: ${data.message_id}`);

      if (data.reply) {
        appendMessage(
          "AI",
          data.reply,
          data.model_deployment_name,
          data.message_id,
          data.augmented,
          data.hybrid_citations,
          data.web_search_citations
        );
      }
      if (data.image_url) {
        appendMessage(
          "image",
          data.image_url,
          data.model_deployment_name,
          data.message_id
        );
      }

      if (data.conversation_id) {
        currentConversationId = data.conversation_id;
        const convoItem = document.querySelector(
          `.conversation-item[data-conversation-id="${currentConversationId}"]`
        );
        if (convoItem) {
          let updated = false;
          if (
            data.conversation_title &&
            convoItem.getAttribute("data-conversation-title") !==
              data.conversation_title
          ) {
            convoItem.setAttribute(
              "data-conversation-title",
              data.conversation_title
            );
            const titleEl = convoItem.querySelector(".conversation-title");
            if (titleEl) titleEl.textContent = data.conversation_title;
            updated = true;
          }
          if (data.classification) {
            const currentClassificationJson =
              convoItem.dataset.classifications || "[]";
            const newClassificationJson = JSON.stringify(data.classification);
            if (currentClassificationJson !== newClassificationJson) {
              convoItem.dataset.classifications = newClassificationJson;
              updated = true;
            }
          }
          const dateEl = convoItem.querySelector("small");
          if (dateEl)
            dateEl.textContent = new Date().toLocaleString([], {
              dateStyle: "short",
              timeStyle: "short",
            });

          if (updated) {
            selectConversation(currentConversationId);
          }
        } else {
          addConversationToList(
            currentConversationId,
            data.conversation_title,
            data.classification || []
          );
          selectConversation(currentConversationId);
        }
      }
    })
    .catch((error) => {
      hideLoadingIndicatorInChatbox();
      console.error("Error sending message:", error);

      if (error.status === 403 && error.data) {
        const categories = (error.data.triggered_categories || [])
          .map((catObj) => `${catObj.category} (severity=${catObj.severity})`)
          .join(", ");
        const reasonMsg = Array.isArray(error.data.reason)
          ? error.data.reason.join(", ")
          : error.data.reason;

        appendMessage(
          "safety",
          `Your message was blocked by Content Safety.\n\n` +
            `**Categories triggered**: ${categories}\n` +
            `**Reason**: ${reasonMsg}`,
          null,
          error.data.message_id
        );
      } else {
        const errMsg = (error.message || "").toLowerCase();
        if (errMsg.includes("embedding") || error.status === 500) {
          appendMessage(
            "Error",
            "There was an issue with the embedding process. Please check with an admin on embedding configuration."
          );
        } else {
          appendMessage(
            "Error",
            `Could not get a response. ${error.message || ""}`
          );
        }
      }
    });
}

/* === NEW: Streaming sender (NDJSON) — true interrupt + citations on done === */
async function actuallySendMessageStream(finalMessageToSend) {
  const modelDeployment = modelSelect?.value;

  appendMessage("You", finalMessageToSend);
  userInput.value = "";
  userInput.style.height = "";
  showLoadingIndicatorInChatbox();
  setInFlightState(true);

  currentChatAbortController = new AbortController();

  let hybridSearchEnabled = false;
  const sdbtn = document.getElementById("search-documents-btn");
  if (sdbtn && sdbtn.classList.contains("active")) {
    hybridSearchEnabled = true;
  }

  let selectedDocumentId = null;
  let classificationsToSend = null;
  const docSel = document.getElementById("document-select");
  const classificationInput = document.getElementById("classification-select");

  if (docSel) {
    const selectedDocOption = docSel.options[docSel.selectedIndex];
    selectedDocumentId =
      selectedDocOption && selectedDocOption.value !== ""
        ? selectedDocOption.value
        : null;
  }

  if (classificationInput) {
    classificationsToSend =
      classificationInput.value === "N/A" ? null : classificationInput.value;
  }

  let bingSearchEnabled = false;
  const wbbtn = document.getElementById("search-web-btn");
  if (wbbtn && wbbtn.classList.contains("active")) {
    bingSearchEnabled = true;
  }

  let imageGenEnabled = false;
  const igbtn = document.getElementById("image-generate-btn");
  if (igbtn && igbtn.classList.contains("active")) {
    imageGenEnabled = true;
  }

  const body = {
    message: finalMessageToSend,
    conversation_id: currentConversationId,
    hybrid_search: hybridSearchEnabled,
    selected_document_id: selectedDocumentId,
    classifications: classificationsToSend,
    bing_search: bingSearchEnabled,
    image_generation: imageGenEnabled,
    doc_scope: docScopeSelect ? docScopeSelect.value : "all",
    active_group_id: window.activeGroupId,
    model_deployment: modelDeployment,
  };

  // Create an empty AI message to stream into
  const aiNode = appendMessage("AI", "", modelDeployment, null, false, [], []);
  const textContainer = aiNode?.querySelector(".message-text");
  const bubble = aiNode?.querySelector(".message-bubble");
  const footer = aiNode?.querySelector(".message-footer");
  const copyBtn = aiNode?.querySelector(".copy-btn");
  const hiddenId = copyBtn?.getAttribute("data-hidden-text-id");
  const hiddenTextarea = hiddenId ? document.getElementById(hiddenId) : null;

  // Accumulate full text for final Markdown render
  let streamedText = "";
  // Track whether we refreshed the left conversation list
  let didRefreshConvos = false; // <-- ADDED

  try {
    const resp = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: currentChatAbortController.signal,
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      pending += decoder.decode(value, { stream: true });

      let nl;
      while ((nl = pending.indexOf("\n")) >= 0) {
        const line = pending.slice(0, nl).trim();
        pending = pending.slice(nl + 1);
        if (!line) continue;

        let evt;
        try {
          evt = JSON.parse(line);
        } catch {
          continue;
        }

        if (evt.type === "meta") {
          // Update conversation / title immediately but DO NOT reload messages during stream
          if (evt.conversation_id) {
            currentConversationId = evt.conversation_id;
          }
          if (evt.conversation_title) {
            const convoItem = document.querySelector(
              `.conversation-item[data-conversation-id="${currentConversationId}"]`
            );
            if (convoItem) {
              convoItem.setAttribute(
                "data-conversation-title",
                evt.conversation_title
              );
              const titleEl = convoItem.querySelector(".conversation-title");
              if (titleEl) titleEl.textContent = evt.conversation_title;
            } else {
              addConversationToList(
                currentConversationId,
                evt.conversation_title,
                evt.classification || []
              );
            }
            // Avoid selectConversation(currentConversationId) here;
            // it would call loadMessages() and wipe the streaming bubble.
          }
        } else if (evt.type === "token") {
          if (evt.token) {
            // ====== NEW: Strip calc-details tag from token before appending ======
            const [cleanToken, hadCalc] = stripCalcDetailsToken(evt.token);
            streamedText += cleanToken;
            // Append as plain text while streaming for performance
            if (textContainer) {
              textContainer.insertAdjacentText("beforeend", cleanToken);
              try {
                safeScrollToBottom();
              } catch {}
            }
            if (hiddenTextarea) {
              hiddenTextarea.value = streamedText;
            }
            // Optionally: handle calc-details inline (if needed per token)
          }
        } else if (evt.type === "image") {
          // Optional: ignore for now
        } else if (evt.type === "error") {
          appendMessage("Error", evt.message || "Streaming failed.");
        } else if (evt.type === "done") {
          // --- collect ids first (so we can safely use them below) ---
          const messageId = evt.message_id || aiNode?.getAttribute("data-message-id") || `msg-${Date.now()}`;
          if (aiNode && !aiNode.getAttribute("data-message-id")) {
            aiNode.setAttribute("data-message-id", messageId);
          }

          // Finalize: render Markdown and insert citations (if any)
          const cleaned = (streamedText || "")
            .trim()
            .replace(/\n{3,}/g, "\n\n")
            .replace(/(\bhttps?:\/\/\S+)(%5D|\])+/gi, (_, url) => url);
          const withInlineCitations = parseCitations(cleaned);
          const finalHtml = DOMPurify.sanitize(marked.parse(withInlineCitations));
          if (textContainer) {
            textContainer.innerHTML = finalHtml; // Replace plain text with formatted HTML
            attachCodeBlockCopyButtons(textContainer);

            // ====== NEW: Render calc-details panel if present and not already rendered ======
            const guardId = messageId;
            if (!detailsRenderedForMsgIds.has(guardId) && streamedText.includes("<calc-details")) {
              renderCalcDetailsFromTokenText(streamedText, bubble);
              detailsRenderedForMsgIds.add(guardId);
            }
          }

          // Inject citations toggle if augmentation present
          const augmented = !!evt.augmented;
          const hybridCitations = evt.hybrid_citations || [];
          const webCitations = evt.web_search_citations || [];

          if (
            augmented &&
            (hybridCitations.length || webCitations.length) &&
            bubble
          ) {
            // Avoid duplicating if already present
            let existingCitations = bubble.querySelector(".citations-container");
            let existingToggle = bubble.querySelector(".citation-toggle-btn");
            if (!existingCitations && !existingToggle) {
              const citationsButtonsHtml = createCitationsHtml(
                hybridCitations,
                webCitations,
                messageId
              );
              const containerId = `citations-${messageId || Date.now()}`;
              const citationContent = document.createElement("div");
              citationContent.className =
                "citations-container mt-2 pt-2 border-top";
              citationContent.id = containerId;
              citationContent.style.display = "none";
              citationContent.innerHTML = citationsButtonsHtml;

              // Place citations block before footer
              if (footer) {
                bubble.insertBefore(citationContent, footer);
              } else {
                bubble.appendChild(citationContent);
              }

              // Add toggle button into footer (create one if footer missing)
              let footerEl = footer;
              if (!footerEl) {
                footerEl = document.createElement("div");
                footerEl.className =
                  "message-footer d-flex justify-content-between align-items-center";
                bubble.appendChild(footerEl);
              }
              const toggleWrap = document.createElement("div");
              toggleWrap.className = "citation-toggle-container";
              toggleWrap.innerHTML = `<button class="btn btn-sm btn-outline-secondary citation-toggle-btn" title="Show sources" aria-expanded="false" aria-controls="${containerId}"><i class="bi bi-journal-text"></i></button>`;
              footerEl.appendChild(toggleWrap);

              const toggleBtn = toggleWrap.querySelector(".citation-toggle-btn");
              toggleBtn?.addEventListener("click", () => {
                const citationsContainer =
                  bubble.querySelector(`#${containerId}`);
                if (!citationsContainer) return;
                const isExpanded =
                  citationsContainer.style.display !== "none";
                citationsContainer.style.display = isExpanded ? "none" : "block";
                toggleBtn.setAttribute("aria-expanded", !isExpanded);
                toggleBtn.title = isExpanded ? "Show sources" : "Hide sources";
                toggleBtn.innerHTML = isExpanded
                  ? '<i class="bi bi-journal-text"></i>'
                  : '<i class="bi bi-chevron-up"></i>';
                if (!isExpanded) {
                  try {
                    safeScrollToBottom();
                  } catch {}
                }
              });
            }
          }

          // *** NEW: Refresh the conversation list once the stream completes ***
          try {
            if (!didRefreshConvos) {
              await loadConversations();
              didRefreshConvos = true;
            }
          } catch (e) {
            console.warn("loadConversations() failed:", e);
          }

          // End request state on 'done'
          setInFlightState(false);
        }
      }
    }
  } catch (error) {
    const wasAborted =
      error &&
      (error.name === "AbortError" || /aborted/i.test(String(error.message)));
    if (wasAborted) {
      showToast("Generation stopped.", "warning");
    } else {
      appendMessage("Error", `Could not get a response. ${error.message || ""}`);
    }
  } finally {
    hideLoadingIndicatorInChatbox();
    setInFlightState(false);
    currentChatAbortController = null;

    // Safety refresh if we never got to 'done'
    try {
      if (typeof didRefreshConvos !== "undefined" && !didRefreshConvos) {
        await loadConversations();
      }
    } catch {}
  }
}

// Helper to append text to assistant message element
function appendToAssistantMessage(msgEl, text) {
  if (!msgEl) return;
  let textContainer = msgEl.querySelector(".message-text");
  if (!textContainer) {
    textContainer = document.createElement("div");
    textContainer.className = "message-text";
    msgEl.appendChild(textContainer);
  }
  textContainer.insertAdjacentText("beforeend", text);
}

// ===== Calc Details (inline helper) =====

// Returns [cleanText, hadCalcPayload]
// Use this BEFORE adding token text to the DOM so the raw tag never shows.
function stripCalcDetailsToken(rawTokenText) {
  if (!rawTokenText) return [rawTokenText, false];
  const m = rawTokenText.match(/<calc-details\s+data-json='([^']+)'>/);
  if (!m) return [rawTokenText, false];
  const cleaned = rawTokenText.replace(m[0], "");
  return [cleaned, true];
}

// Renders the collapsible panel once per assistant message.
// Call this AFTER the message element exists in the DOM.
function renderCalcDetailsFromTokenText(tokenText, mountEl) {
  if (!tokenText || !mountEl) return false;
  const m = tokenText.match(/<calc-details\s+data-json='([^']+)'>/);
  if (!m) return false;

  // Parse JSON payload stored in the data attribute
  let payload;
  try {
    const jsonStr = m[1].replace(/&apos;/g, "'"); // backend encodes ' as &apos;
    payload = JSON.parse(jsonStr);
  } catch (e) {
    console.warn("[calc-details] Failed to parse payload:", e);
    return false;
  }

  // <details> container
  const details = document.createElement("details");
  details.className = "calc-details mt-2";
  details.open = false;
  const summary = document.createElement("summary");
  summary.textContent = "Calculation details";
  details.appendChild(summary);

  const structured = payload.structured || {};
  // Optional: pretty table for grouped results
  if (Array.isArray(structured.group_aggregate) && structured.group_aggregate.length) {
    const metric = ["mean", "sum", "min", "max", "median"].find(k => k in structured.group_aggregate[0]) || "mean";
    const table = document.createElement("table");
    table.className = "table table-sm table-striped mt-2";
    table.innerHTML = "<thead><tr><th>Group</th><th>n</th><th>Value</th></tr></thead><tbody></tbody>";
    const tbody = table.querySelector("tbody");
    structured.group_aggregate.slice(0, 50).forEach(row => {
      const tr = document.createElement("tr");
      const tdG = document.createElement("td"); tdG.textContent = row.group;
      const tdN = document.createElement("td"); tdN.textContent = (row.n ?? "").toString();
      const tdV = document.createElement("td");
      const v = row[metric];
      tdV.textContent = (typeof v === "number") ? v.toLocaleString() : (v ?? "").toString();
      tr.appendChild(tdG); tr.appendChild(tdN); tr.appendChild(tdV);
      tbody.appendChild(tr);
    });
    details.appendChild(table);
  }

  // Raw JSON fallback (audit + structured)
  const pre = document.createElement("pre");
  pre.style.whiteSpace = "pre-wrap";
  pre.style.fontSize = "0.85rem";
  pre.textContent = JSON.stringify(payload, null, 2);
  details.appendChild(pre);

  // Append under the assistant message
  mountEl.appendChild(details);
  return true;
}

// Keep track so we don’t render the panel twice if multiple tokens contain the tag
const detailsRenderedForMsgIds = new Set();

function attachCodeBlockCopyButtons(parentElement) {
  if (!parentElement) return;
  const codeBlocks = parentElement.querySelectorAll("pre code");
  codeBlocks.forEach((codeBlock) => {
    const pre = codeBlock.parentElement;
    if (pre.querySelector(".copy-code-btn")) return;

    pre.style.position = "relative";
    const copyBtn = document.createElement("button");
    copyBtn.innerHTML = '<i class="bi bi-copy"></i>';
    copyBtn.classList.add(
      "copy-code-btn",
      "btn",
      "btn-sm",
      "btn-outline-secondary"
    );
    copyBtn.title = "Copy code";
    copyBtn.style.position = "absolute";
    copyBtn.style.top = "5px";
    copyBtn.style.right = "5px";
    copyBtn.style.lineHeight = "1";
    copyBtn.style.padding = "0.15rem 0.3rem";

    copyBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const codeToCopy = codeBlock.innerText;
      navigator.clipboard
        .writeText(codeToCopy)
        .then(() => {
          copyBtn.innerHTML = '<i class="bi bi-check-lg text-success"></i>';
          copyBtn.title = "Copied!";
          setTimeout(() => {
            copyBtn.innerHTML = '<i class="bi bi-copy"></i>';
            copyBtn.title = "Copy code";
          }, 2000);
        })
        .catch((err) => {
          console.error("Error copying code:", err);
          showToast("Failed to copy code.", "warning");
        });
    });
    pre.appendChild(copyBtn);
  });
}

if (sendBtn) {
  sendBtn.addEventListener("click", () => {
    if (isRequestInFlight) {
      try {
        currentChatAbortController?.abort();
      } catch {}
      hideLoadingIndicatorInChatbox();
      setInFlightState(false);
      return;
    }
    sendMessage();
  });
}

if (userInput) {
  userInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter") {
      if (!e.shiftKey) {
        e.preventDefault();
        if (isRequestInFlight) {
          try {
            currentChatAbortController?.abort();
          } catch {}
          hideLoadingIndicatorInChatbox();
          setInFlightState(false);
          return;
        }
        sendMessage();
      }
    }
  });
}

if (modelSelect) {
  modelSelect.addEventListener("change", function () {
    const selectedModel = modelSelect.value;
    console.log(`Saving preferred model: ${selectedModel}`);
    saveUserSetting({ preferredModelDeployment: selectedModel });
  });
}

/* ===========================
   Next-word Ghost Suggestion
   =========================== */
(() => {
  const inputEl = userInput;
  const overlayEl = document.getElementById("ghost-overlay");

  if (!inputEl || !overlayEl) return;

  let debounceId = null;
  let currentSuggestion = "";
  let lastPrefixSent = "";
  const DEBOUNCE_MS = 250;

  const atEnd = (el) =>
    el.selectionStart === el.value.length && el.selectionEnd === el.value.length;

  function renderOverlay(prefix, suggestion) {
    const typedSpan = `<span class="typed" style="color: transparent;">${escapeHtml(
      prefix
    )}</span>`;
    const sugSpan = suggestion
      ? ` <span class="suggestion" style="color:#9aa0a6;">${escapeHtml(
          suggestion
        )}</span>`
      : "";
    overlayEl.innerHTML = typedSpan + sugSpan;
    overlayEl.style.visibility = prefix || suggestion ? "visible" : "hidden";
  }

  async function fetchSuggestion(prefix) {
    try {
      const resp = await fetch("/api/predict-next-word", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prefix }),
      });
      if (!resp.ok) return "";
      const data = await resp.json();
      return typeof data?.suggestion === "string" ? data.suggestion : "";
    } catch {
      return "";
    }
  }

  function shouldQuery(prefix) {
    if (!prefix || prefix.length < 3) return false;
    if (!/\w$/.test(prefix)) return false;
    if (prefix === lastPrefixSent) return false;
    return true;
  }

  function clearSuggestion() {
    currentSuggestion = "";
    renderOverlay(inputEl.value || "", "");
  }

  function maybeQuery() {
    const prefix = inputEl.value || "";
    if (!atEnd(inputEl) || !shouldQuery(prefix)) {
      currentSuggestion = "";
      renderOverlay(prefix, "");
      return;
    }

    clearTimeout(debounceId);
    debounceId = setTimeout(async () => {
      lastPrefixSent = prefix;
      const suggestion = await fetchSuggestion(prefix);
      currentSuggestion = suggestion || "";
      renderOverlay(prefix, currentSuggestion);
    }, DEBOUNCE_MS);
  }

  inputEl.addEventListener("input", maybeQuery);

  inputEl.addEventListener("keyup", (e) => {
    if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) {
      if (!atEnd(inputEl)) currentSuggestion = "";
      renderOverlay(
        inputEl.value || "",
        atEnd(inputEl) ? currentSuggestion : ""
      );
    }
  });
  inputEl.addEventListener("click", () => {
    renderOverlay(inputEl.value || "", atEnd(inputEl) ? currentSuggestion : "");
  });
  inputEl.addEventListener("mouseup", () => {
    renderOverlay(inputEl.value || "", atEnd(inputEl) ? currentSuggestion : "");
  });

  inputEl.addEventListener("keydown", (e) => {
    if (
      currentSuggestion &&
      atEnd(inputEl) &&
      (e.key === "Tab" || e.key === "ArrowRight")
    ) {
      e.preventDefault();
      inputEl.value = (inputEl.value || "") + currentSuggestion;
      currentSuggestion = "";
      renderOverlay(inputEl.value, "");
      inputEl.dispatchEvent(new Event("input", { bubbles: true }));
      return;
    }
    if (e.key === "Escape") {
      clearSuggestion();
      return;
    }
    if (e.key === "Enter" && !e.shiftKey) {
      clearSuggestion();
    }
  });

  if (sendBtn) {
    sendBtn.addEventListener("click", clearSuggestion);
  }

  renderOverlay(inputEl.value || "", "");
})();
