"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ShieldAlert, LayoutDashboard, Users, Activity, Settings, Bell } from "lucide-react";
import { cn } from "@/lib/utils";

const routes = [
  {
    label: "Executive Overview",
    icon: LayoutDashboard,
    href: "/",
    color: "text-sky-500",
  },
  {
    label: "Suspicious Accounts",
    icon: Users,
    href: "/accounts",
    color: "text-violet-500",
  },
  {
    label: "Alerts Center",
    icon: Bell,
    href: "/alerts",
    color: "text-pink-700",
  },
  {
    label: "Transactions",
    icon: Activity,
    href: "/transactions",
    color: "text-emerald-500",
  }
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="space-y-4 py-4 flex flex-col h-full bg-slate-900 w-64 border-r border-border text-white">
      <div className="px-3 py-2 flex-1">
        <Link href="/" className="flex items-center pl-3 mb-14">
          <div className="relative w-8 h-8 mr-4 text-primary">
            <ShieldAlert size={32} className="text-blue-500" />
          </div>
          <h1 className="text-xl font-bold">
            Fraud<span className="text-blue-500">Intel</span>
          </h1>
        </Link>
        <div className="space-y-1">
          {routes.map((route) => (
            <Link
              key={route.href}
              href={route.href}
              className={cn(
                "text-sm group flex p-3 w-full justify-start font-medium cursor-pointer hover:text-white hover:bg-white/10 rounded-lg transition",
                pathname === route.href || (pathname.startsWith(route.href) && route.href !== "/") 
                  ? "text-white bg-white/10"
                  : "text-zinc-400"
              )}
            >
              <div className="flex items-center flex-1">
                <route.icon className={cn("h-5 w-5 mr-3", route.color)} />
                {route.label}
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
