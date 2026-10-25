import nextVitals from 'eslint-config-next/core-web-vitals';

const config = [
  {
    ignores: ['.next/**', 'node_modules/**', 'out/**', 'next-env.d.ts'],
  },
  ...nextVitals,
  {
    rules: {
      // These pages intentionally hydrate local state or reset state when the
      // selected result changes; the rule is a React Compiler recommendation,
      // not a correctness requirement here.
      'react-hooks/set-state-in-effect': 'off',
    },
  },
];

export default config;
