// chat-feedback.js

import { showToast } from "./chat-toast.js";
import { toBoolean } from "./chat-utils.js";

const feedbackForm = document.getElementById("feedback-form");

export function renderFeedbackIcons(messageId, conversationId) {
  if (toBoolean(window.enableUserFeedback)) {
    return `
      <div class="feedback-icons" data-ai-message-id="${messageId}">
        <i class="bi bi-hand-thumbs-up-fill text-muted me-3 feedback-btn" 
          data-feedback-type="positive" 
          data-conversation-id="${conversationId}"
          title="Thumbs Up"
          style="cursor:pointer;"></i>
        <i class="bi bi-hand-thumbs-down-fill text-muted feedback-btn" 
          data-feedback-type="negative" 
          data-conversation-id="${conversationId}"
          title="Thumbs Down"
          style="cursor:pointer;"></i>
      </div>
    `;
  }
  else {
    return "";
  }
}

export function submitFeedback(messageId, conversationId, feedbackType, reason) {
  fetch("/feedback/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messageId,
      conversationId,
      feedbackType,
      reason
    }),
  })
    .then((resp) => resp.json())
    .then((data) => {
      if (data.success) {
        console.log("Feedback submitted:", data);
      } else {
        console.error("Feedback error:", data.error || data);
        showToast("Error submitting feedback: " + (data.error || "Unknown error."), "danger");
      }
    })
    .catch((err) => {
      console.error("Error sending feedback:", err);
      showToast("Error sending feedback.", "danger");
    });
}

document.addEventListener("click", function (event) {
  const feedbackBtn = event.target.closest(".feedback-btn");
  if (!feedbackBtn) return;

  const feedbackType = feedbackBtn.getAttribute("data-feedback-type");
  const messageId = feedbackBtn.closest(".feedback-icons").getAttribute("data-ai-message-id");
  const conversationId = feedbackBtn.getAttribute("data-conversation-id");

  feedbackBtn.classList.add("clicked");

  if (feedbackType === "positive") {
    submitFeedback(messageId, conversationId, "positive", "");

    setTimeout(() => {
      feedbackBtn.classList.remove("clicked");
    }, 500);
  } else {
    const modalEl = new bootstrap.Modal(document.getElementById("feedback-modal"));
    document.getElementById("feedback-ai-response-id").value = messageId;
    document.getElementById("feedback-conversation-id").value = conversationId;
    document.getElementById("feedback-type").value = "negative";
    document.getElementById("feedback-reason").value = "";
    modalEl.show();
  }
});

if (feedbackForm) {
  feedbackForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const messageId = document.getElementById("feedback-ai-response-id").value;
    const conversationId = document.getElementById("feedback-conversation-id").value;
    const feedbackType = document.getElementById("feedback-type").value;
    const reason = document.getElementById("feedback-reason").value.trim();

    submitFeedback(messageId, conversationId, feedbackType, reason);

    const modalEl = bootstrap.Modal.getInstance(
      document.getElementById("feedback-modal")
    );
    if (modalEl) modalEl.hide();
  });
}

