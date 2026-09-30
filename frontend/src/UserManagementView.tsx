import { FormEvent, useEffect, useState } from "react";
import { Shield, UserPlus, Users } from "lucide-react";
import {
  createUser,
  deleteUser,
  listUsers,
  updateUser,
  type AuthUser,
  type UserRole,
} from "./api";
import { roleLabel, useAuth } from "./auth";

export function UserManagementView() {
  const { user: currentUser } = useAuth();
  const [users, setUsers] = useState<AuthUser[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<UserRole>("annotator");
  const [busy, setBusy] = useState(false);

  async function reload() {
    const response = await listUsers();
    setUsers(response.items);
  }

  useEffect(() => {
    void reload().catch((err) => {
      setError(err instanceof Error ? err.message : "加载用户失败");
    });
  }, []);

  async function handleCreate(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await createUser({ username: username.trim(), password, role });
      setUsername("");
      setPassword("");
      setRole("annotator");
      await reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : "创建用户失败");
    } finally {
      setBusy(false);
    }
  }

  async function handleToggleActive(target: AuthUser) {
    setBusy(true);
    setError(null);
    try {
      await updateUser(target.id, { is_active: !target.is_active });
      await reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : "更新用户失败");
    } finally {
      setBusy(false);
    }
  }

  async function handleRoleChange(target: AuthUser, nextRole: UserRole) {
    setBusy(true);
    setError(null);
    try {
      await updateUser(target.id, { role: nextRole });
      await reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : "更新角色失败");
    } finally {
      setBusy(false);
    }
  }

  async function handleResetPassword(target: AuthUser) {
    const next = window.prompt(`为 ${target.username} 设置新密码（至少 6 位）`);
    if (!next) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await updateUser(target.id, { password: next });
      await reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : "重置密码失败");
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(target: AuthUser) {
    if (!window.confirm(`确认删除用户 ${target.username}？`)) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await deleteUser(target.id);
      await reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : "删除用户失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel user-management-panel" aria-label="用户管理">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">权限与账号</p>
          <h2>用户管理</h2>
          <p className="muted">管理员可创建标注员、调整角色与启停账号</p>
        </div>
        <Users size={22} />
      </div>

      {error ? <div className="error-banner">{error}</div> : null}

      <form className="user-create-form" onSubmit={(event) => void handleCreate(event)}>
        <div className="panel-heading compact">
          <div>
            <p className="eyebrow">新建用户</p>
            <h3>添加管理员或标注员</h3>
          </div>
          <UserPlus size={18} />
        </div>
        <div className="user-create-grid">
          <label>
            用户名
            <input
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              minLength={2}
              required
            />
          </label>
          <label>
            初始密码
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              minLength={6}
              required
            />
          </label>
          <label>
            角色
            <select value={role} onChange={(event) => setRole(event.target.value as UserRole)}>
              <option value="annotator">标注员</option>
              <option value="admin">管理员</option>
            </select>
          </label>
          <button type="submit" disabled={busy}>
            创建用户
          </button>
        </div>
      </form>

      <div className="user-table-wrap">
        <table className="user-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>用户名</th>
              <th>角色</th>
              <th>状态</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            {users.map((item) => {
              const isSelf = item.id === currentUser?.id;
              return (
                <tr key={item.id}>
                  <td>{item.id}</td>
                  <td>
                    <span className="user-name-cell">
                      <Shield size={14} />
                      {item.username}
                      {isSelf ? <span className="pill soft">当前</span> : null}
                    </span>
                  </td>
                  <td>
                    <select
                      value={item.role}
                      disabled={busy || isSelf}
                      onChange={(event) =>
                        void handleRoleChange(item, event.target.value as UserRole)
                      }
                    >
                      <option value="admin">{roleLabel("admin")}</option>
                      <option value="annotator">{roleLabel("annotator")}</option>
                    </select>
                  </td>
                  <td>{item.is_active ? "启用" : "停用"}</td>
                  <td className="user-actions">
                    <button
                      type="button"
                      className="secondary-button"
                      disabled={busy || isSelf}
                      onClick={() => void handleToggleActive(item)}
                    >
                      {item.is_active ? "停用" : "启用"}
                    </button>
                    <button
                      type="button"
                      className="secondary-button"
                      disabled={busy}
                      onClick={() => void handleResetPassword(item)}
                    >
                      重置密码
                    </button>
                    <button
                      type="button"
                      className="secondary-button danger"
                      disabled={busy || isSelf}
                      onClick={() => void handleDelete(item)}
                    >
                      删除
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
