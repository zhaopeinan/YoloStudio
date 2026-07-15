import type { ReactNode } from "react";
import type { WorkflowStep, WorkflowStepItem } from "./workflow";

type WorkflowShellProps = {
  currentStep: WorkflowStep;
  steps: WorkflowStepItem[];
  navigationNotice: string | null;
  onNavigate: (step: WorkflowStep) => void;
  children: ReactNode;
};

function formatStepStatus(step: WorkflowStepItem): string {
  if (step.availability === "locked") {
    return "暂不可用";
  }
  if (step.status === "completed") {
    return "已完成";
  }
  if (step.status === "in_progress") {
    return "进行中";
  }
  return "未开始";
}

export function WorkflowShell({
  currentStep,
  steps,
  navigationNotice,
  onNavigate,
  children,
}: WorkflowShellProps) {
  const currentIndex = steps.findIndex((step) => step.id === currentStep);
  const previousStep = currentIndex > 0 ? steps[currentIndex - 1] : null;
  const nextStep = currentIndex >= 0 && currentIndex < steps.length - 1
    ? steps[currentIndex + 1]
    : null;

  return (
    <div className="workflow-shell">
      <aside className="workflow-sidebar">
        <nav aria-label="工作流步骤" className="workflow-navigation">
          {steps.map((step) => (
            <button
              type="button"
              key={step.id}
              className={step.id === currentStep ? "workflow-step active" : "workflow-step"}
              aria-current={step.id === currentStep ? "step" : undefined}
              aria-disabled={step.availability === "locked"}
              title={step.lockedReason}
              onClick={() => onNavigate(step.id)}
            >
              <span className="workflow-step-number">{step.number}</span>
              <span className="workflow-step-copy">
                <strong>{step.label}</strong>
                <small>{step.description}</small>
              </span>
              <span className={`workflow-step-status ${step.status}`}>
                {formatStepStatus(step)}
              </span>
            </button>
          ))}
        </nav>
      </aside>

      <div className="workflow-mobile-picker">
        <label htmlFor="workflow-step-select">当前步骤</label>
        <select
          id="workflow-step-select"
          value={currentStep}
          onChange={(event) => onNavigate(event.target.value as WorkflowStep)}
        >
          {steps.map((step) => (
            <option key={step.id} value={step.id}>
              {step.number}. {step.label} · {step.availability === "locked" ? "暂不可用" : "可进入"}
            </option>
          ))}
        </select>
      </div>

      <section className="workflow-content" aria-label="当前工作流页面">
        {navigationNotice ? <div className="warning-banner">{navigationNotice}</div> : null}
        {children}
        {currentStep !== "annotation" ? (
          <footer className="workflow-page-actions">
            {previousStep ? (
              <button
                type="button"
                className="secondary-button"
                onClick={() => onNavigate(previousStep.id)}
              >
                上一步：{previousStep.label}
              </button>
            ) : (
              <span />
            )}
            {nextStep ? (
              <button type="button" onClick={() => onNavigate(nextStep.id)}>
                下一步：{nextStep.label}
              </button>
            ) : null}
          </footer>
        ) : null}
      </section>
    </div>
  );
}
