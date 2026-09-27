/**
 * script.js — Fake News Detector
 * Simple client-side validation for the prediction form.
 */

document.addEventListener("DOMContentLoaded", function () {
    // Get the prediction form
    var form = document.getElementById("predict-form");

    if (form) {
        form.addEventListener("submit", function (event) {
            var headline = document.getElementById("headline");

            // Check if the textarea is empty
            if (headline && headline.value.trim() === "") {
                event.preventDefault();
                alert("Please enter a news headline.");
                headline.focus();
            }
        });
    }
});
