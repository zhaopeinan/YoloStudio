import { FormEvent, useState } from "react";
import { LogIn } from "lucide-react";
import { useAuth } from "./auth";

export function LoginView() {
  const { login } = useAuth();
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await login(username.trim(), password);
    } catch (err) {
      setError(err instanceof Error ? err.message : "登录失败");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="login-shell">
      <section className="login-card" aria-label="用户登录">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">YOLO Trainer</p>
            <h1>登录工作台</h1>
            <p className="muted">使用管理员或标注员账号进入系统</p>
          </div>
          <LogIn size={22} />
        </div>

        <form className="login-form" onSubmit={(event) => void handleSubmit(event)}>
          <label>
            用户名
            <input
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              autoComplete="username"
              required
            />
          </label>
          <label>
            密码
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="current-password"
              required
            />
          </label>
          {error ? <div className="error-banner">{error}</div> : null}
          <button type="submit" disabled={submitting}>
            {submitting ? "登录中…" : "登录"}
          </button>
        </form>
      </section>
    </main>
  );
}
