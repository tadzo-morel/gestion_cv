import { render, screen } from '@testing-library/react';
import App from './App';

beforeEach(() => {
  global.fetch = jest.fn(() =>
    Promise.resolve({
      ok: true,
      json: () => Promise.resolve({
        total_cvs: 2483,
        categories: 24,
        model_accuracy: 70.42,
        model_name: 'Random Forest',
      }),
    })
  );
});

afterEach(() => {
  jest.resetAllMocks();
});

test('affiche le titre et les deux onglets', async () => {
  render(<App />);
  expect(screen.getByText(/dossier cv/i)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /^analyser$/i })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /^comparer$/i })).toBeInTheDocument();
  expect(await screen.findByText(/2483/)).toBeInTheDocument();
});
