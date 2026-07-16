import { EpochAdminApp } from "@/components/admin/EpochAdminApp";

export const metadata = {
  title: "AgentForge — Epoch Admin (RQGM 進化控制台)",
  description:
    "RQGM 進化控制台：兩段式晉升門檻（code gate 先行、HITL 為否決安全鎖）、Pareto frontier population 搜尋、對抗/剝削偵測與 val/test 分離度透明報告。",
};

export default function AdminPage() {
  return <EpochAdminApp />;
}
