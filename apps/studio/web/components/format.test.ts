import { describe, expect, it } from 'vitest';
import { significant } from './format';

describe('numbers to a few significant digits', () => {
  it('are written plainly, or with an exponent when very small or large', () => {
    expect(significant(0.0649123, 3)).toBe('0.0649');
    expect(significant(1234.5, 3)).toBe('1230');
    expect(significant(4.9135e-6, 3)).toBe('4.91e-6');
    expect(significant(-0.0005, 3)).toBe('-5e-4');
    expect(significant(2.5e7, 3)).toBe('2.5e7');
    expect(significant(0, 3)).toBe('0');
  });
});
