import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  Activity, 
  Map, 
  Settings, 
  TrendingUp, 
  FolderOpen,
  LayoutDashboard, 
  GitMerge, 
  Layers, 
  RefreshCw,
  Cpu,
  Compass,
  Tag
} from 'lucide-react';

interface AppShellProps {
  children: React.ReactNode;
}

export default function AppShell({ children }: AppShellProps) {
  const location = useLocation();
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [isRefreshing, setIsRefreshing] = useState(false);

  const checkConnection = async () => {
    setIsRefreshing(true);
    try {
      const response = await fetch('/api/health');
      if (response.ok) {
        setBackendStatus('connected');
      } else {
        setBackendStatus('disconnected');
      }
    } catch {
      setBackendStatus('disconnected');
    } finally {
      setTimeout(() => setIsRefreshing(false), 500);
    }
  };

  useEffect(() => {
    checkConnection();
    const interval = setInterval(checkConnection, 10000);
    return () => clearInterval(interval);
  }, []);

  const menuItems = [
    { label: 'System Overview', path: '/', icon: LayoutDashboard },
    { label: 'Analysis Workspace', path: '/workspace', icon: Map },
    { label: 'New Route Discovery', path: '/discovery', icon: Compass },
    { label: 'Road Classification', path: '/classification', icon: Tag },
    { label: 'Network Intelligence', path: '/intelligence', icon: GitMerge },
    { label: 'Comparison Mode', path: '/comparison', icon: Layers },
  ];

  return (
    <div className="flex h-screen bg-[#070a0e] text-[#c9d1d9] font-sans overflow-hidden">
      {/* Sidebar Navigation */}
      <aside className="w-64 border-r border-[#1f242c] bg-[#0b0f14] flex flex-col justify-between shrink-0">
        <div>
          {/* Header */}
          <div className="h-16 border-b border-[#1f242c] flex items-center px-6 gap-3">
            <div className="bg-emerald-500/10 p-2 rounded border border-emerald-500/25">
              <Activity className="h-5 w-5 text-emerald-400" />
            </div>
            <div>
              <div className="font-bold text-white tracking-wider text-sm font-mono">ROADINTEL</div>
              <div className="text-[10px] text-gray-500 font-medium uppercase tracking-widest">GeoAI Engine</div>
            </div>
          </div>

          {/* Nav Items */}
          <nav className="p-4 space-y-1">
            {menuItems.map((item) => {
              const active = location.pathname === item.path;
              const Icon = item.icon;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center gap-3 px-4 py-2.5 rounded text-sm transition-all duration-150 ${
                    active 
                      ? 'bg-emerald-500/10 border-l-2 border-emerald-500 text-white font-medium' 
                      : 'text-gray-400 hover:text-white hover:bg-gray-800/30'
                  }`}
                >
                  <Icon className={`h-4.5 w-4.5 ${active ? 'text-emerald-400' : 'text-gray-500'}`} />
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Footer info */}
        <div className="p-4 border-t border-[#1f242c] text-xs text-gray-500 space-y-1">
          <div>Study Area: Meridian Region</div>
          <div>CRS Coordinate: EPSG:4326</div>
          <div>Vite Build: v1.0.0 (TS/Tailwind)</div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Header Bar */}
        <header className="h-16 border-b border-[#1f242c] bg-[#0b0f14] flex items-center justify-between px-8 z-10">
          <div>
            <h1 className="text-sm font-semibold text-white tracking-wide font-mono uppercase">
              {menuItems.find((item) => item.path === location.pathname)?.label || 'Workspace'}
            </h1>
          </div>

          {/* System Health / Status Indicators */}
          <div className="flex items-center gap-4 text-xs font-mono">
            {/* Backend Connection */}
            <div className="flex items-center gap-2 px-3 py-1.5 rounded bg-gray-900 border border-[#1f242c]">
              <span className="text-gray-500">API:</span>
              <span className="flex items-center gap-1.5 font-semibold">
                <span className={`h-2 w-2 rounded-full ${
                  backendStatus === 'checking' ? 'bg-amber-400 animate-pulse' :
                  backendStatus === 'connected' ? 'bg-emerald-400' : 'bg-red-400'
                }`} />
                {backendStatus === 'checking' ? 'SYNC' :
                 backendStatus === 'connected' ? 'ONLINE' : 'OFFLINE'}
              </span>
              <button 
                onClick={checkConnection}
                disabled={isRefreshing}
                className="text-gray-500 hover:text-white ml-1 transition-colors"
                title="Refresh Status"
              >
                <RefreshCw className={`h-3 w-3 ${isRefreshing ? 'animate-spin' : ''}`} />
              </button>
            </div>

            {/* Model Status */}
            <div className="flex items-center gap-2 px-3 py-1.5 rounded bg-gray-900 border border-[#1f242c]">
              <Cpu className="h-3.5 w-3.5 text-gray-500" />
              <span className="text-gray-500">MODEL:</span>
              <span className="text-emerald-400 font-semibold">U-NET + Classical</span>
            </div>
          </div>
        </header>

        {/* Content Viewport */}
        <main className="flex-1 overflow-hidden relative">
          {children}
        </main>
      </div>
    </div>
  );
}
