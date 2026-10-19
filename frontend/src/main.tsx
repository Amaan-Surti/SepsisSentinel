import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { DisclaimerBanner, DisclaimerFooter } from "./components/Disclaimer";
import { TopBar } from "./components/TopBar";
import { ModelPage } from "./pages/ModelPage";
import { PatientPage } from "./pages/PatientPage";
import { WardPage } from "./pages/WardPage";
import { ReplayProvider, useReplay } from "./state/ReplayContext";
import "./styles.css";

function Shell() {
  const { configError, retryConfig } = useReplay();
  return (
    <>
      <DisclaimerBanner />
      <TopBar />
      {configError ? (
        <div className="state error">
          {configError} <button onClick={retryConfig}>Retry</button>
        </div>
      ) : (
        <Routes>
          <Route path="/" element={<WardPage />} />
          <Route path="/patient/:id" element={<PatientPage />} />
          <Route path="/model" element={<ModelPage />} />
          <Route path="*" element={<div className="state">Page not found.</div>} />
        </Routes>
      )}
      <DisclaimerFooter />
    </>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <ReplayProvider>
        <Shell />
      </ReplayProvider>
    </BrowserRouter>
  </React.StrictMode>,
);
