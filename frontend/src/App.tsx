import { BrowserRouter, Routes, Route } from 'react-router-dom';
import AppShell from './components/AppShell';
import LandingOverview from './pages/LandingOverview';
import AnalysisWorkspace from './pages/AnalysisWorkspace';
import NewRouteDiscovery from './pages/NewRouteDiscovery';
import RoadClassification from './pages/RoadClassification';
import NetworkIntelligence from './pages/NetworkIntelligence';
import ComparisonMode from './pages/ComparisonMode';

function App() {
  return (
    <BrowserRouter>
      <AppShell>
        <Routes>
          <Route path="/" element={<LandingOverview />} />
          <Route path="/workspace" element={<AnalysisWorkspace />} />
          <Route path="/discovery" element={<NewRouteDiscovery />} />
          <Route path="/classification" element={<RoadClassification />} />
          <Route path="/intelligence" element={<NetworkIntelligence />} />
          <Route path="/comparison" element={<ComparisonMode />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  );
}

export default App;
