/**
 * frontend/src/index.js
 *
 * React entry point. Mounts the App component into #root.
 * Bootstrap CSS loaded here so it's available everywhere.
 */

import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

// Bootstrap for styling — matches what the existing static site uses
import "bootstrap/dist/css/bootstrap.min.css";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
