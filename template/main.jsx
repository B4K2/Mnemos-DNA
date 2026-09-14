import React from "react";
import { createRoot } from "react-dom/client";
import DNAModelComparison from "./dna_model_comparison.jsx";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <DNAModelComparison />
  </React.StrictMode>
);
