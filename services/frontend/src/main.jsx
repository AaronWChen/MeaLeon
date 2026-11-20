/**
 * frontend/src/main.jsx
 *
 * Vite entry point — replaces index.js.
 * Same content, just renamed/relocated per Vite convention
 * (Vite looks for the entry referenced in index.html's <script> tag).
 */

import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

import "bootstrap/dist/css/bootstrap.min.css";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
