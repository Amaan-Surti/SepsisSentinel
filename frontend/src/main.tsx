import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes, useParams } from "react-router-dom";
import { DisclaimerBanner, DisclaimerFooter } from "./components/Disclaimer";
import { TopBar } from "./components/TopBar";
import { TourRail } from "./components/TourRail";
import { BedPage } from "./pages/BedPage";
import { EvidencePage } from "./pages/EvidencePage";
import { TourPage } from "./pages/TourPage";
import { WardPage } from "./pages/WardPage";
import { ReplayProvider, useReplay } from "./state/ReplayContext";
import "./styles.css";

/** Old P7–P9 links (/patient/:id) keep working. */
function LegacyPatient() {
  const { id = "" } = useParams();
  return <Navigate to={`/bed/${id}`} replace />;
}

function Shell() {
  const { configError, retryConfig } = useReplay();
  return (
    <>
      <div className="app-header">
        <DisclaimerBanner />
        <TopBar />
      </div>
      {configError ? (
        <div className="state error" role="alert">
          {configError} <button onClick={retryConfig}>Retry</button>
        </div>
      ) : (
        <Routes>
          <Route path="/" element={<WardPage />} />
          <Route path="/bed/:id" element={<BedPage />} />
          <Route path="/tour" element={<TourPage />} />
          <Route path="/evidence" element={<EvidencePage />} />
          <Route path="/patient/:id" element={<LegacyPatient />} />
          <Route path="/model" element={<Navigate to="/evidence" replace />} />
          <Route path="*" element={<div className="state">Page not found.</div>} />
        </Routes>
      )}
      <DisclaimerFooter />
      <TourRail />
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
