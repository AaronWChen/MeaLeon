import { SearchBar } from './components/SearchBar';

function App() {
  return (
    <div className="container py-4">
      <h1 className="mb-1">MeaLeon</h1>
      <p className="text-muted mb-4">The similar recipe suggester</p>
      <SearchBar />
    </div>
  );
}

export default App;
