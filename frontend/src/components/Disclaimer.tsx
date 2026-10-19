export function DisclaimerBanner() {
  return (
    <div className="disclaimer-banner" role="note">
      <strong>Research demonstration</strong> on retrospective ICU data (PhysioNet 2019). Not approved for clinical
      diagnostic use.
    </div>
  );
}

export function DisclaimerFooter() {
  return (
    <footer className="disclaimer-footer">
      Decision support on retrospective data. Not a medical device. Replay of de-identified lockbox-test patients;
      risks come from the frozen final model.
    </footer>
  );
}
