import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

export const comptoxApiSlice = createApi({
  reducerPath: 'api',
  baseQuery: fetchBaseQuery({ baseUrl: 'http://localhost:3000' }),
  endpoints: (builder) => ({
    fetchConfig: builder.query({
      query: () => `/config`,
    }),
    searchNodes: builder.query({
      // query: (label, field, value) => `/nodes/${label}/search?field=${field}&value=${value}`,
      query: (label) => `/nodes/${label[0]}/search?field=${label[1]}&value=${label[2]}`
      // query: (fields) => `/nodes/${fields}`,
    }),
    fetchRelationshipsByNodeId: builder.query({
      query: (nodeId) => `/relationships/fromStartNodeId/${nodeId}`
    })
  })
});

export const { useFetchConfigQuery, useSearchNodesQuery, useFetchRelationshipsByNodeIdQuery } = comptoxApiSlice;
