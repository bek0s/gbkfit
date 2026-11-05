import { expect, test } from 'vitest';
import { firstSentence } from './schema';

test('the first sentence ends at a period before a capital or a bracket', () => {
  expect(firstSentence('A disk of rings. It is evaluated at every pixel.')).toBe('A disk of rings.');
  expect(firstSentence('A Gaussian (round). (Its sigma.)')).toBe('A Gaussian (round).');
  expect(firstSentence('One sentence only.')).toBe('One sentence only.');
  expect(firstSentence('No period at all')).toBe('No period at all');
});

test('the periods of abbreviations do not end the first sentence', () => {
  expect(firstSentence('The curve of Epinat et al. (2008). More.')).toBe('The curve of Epinat et al. (2008).');
  expect(firstSentence('Bins of pixels (e.g. Voronoi bins). More.')).toBe('Bins of pixels (e.g. Voronoi bins).');
  expect(firstSentence('A shape, i.e. A thing. More.')).toBe('A shape, i.e. A thing.');
});
