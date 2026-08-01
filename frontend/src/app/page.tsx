'use client';

import { useEffect, useState, ReactNode } from 'react'
type Rating = Record<string, unknown>
import { supabase } from '../lib/supabase'

export default function Home() {
  const [ratings, setRatings] = useState<Rating[]>([])

  useEffect(() => {
    const fetchData = async () => {
      const { data, error } = await supabase
        .from<'team_ratings', Rating>('team_ratings')
        .select('*')
      if (error) {
        console.error('Supabase error:', error)
      } else {
        setRatings(data ?? [])
      }
    }

    fetchData()
  }, [])

  return (
    <main className="p-6">
      <h1 className="text-2xl font-bold mb-4">Team Ratings</h1>
      <table className="min-w-full border border-gray-300">
        <thead>
          <tr>
            {ratings[0] &&
              Object.keys(ratings[0]).map((key) => (
                <th key={key} className="border px-4 py-2 text-left bg-gray-100">
                  {key}
                </th>
              ))}
          </tr>
        </thead>
        <tbody>
          {ratings.map((team, index) => (
            <tr key={index}>
              {Object.values(team).map((value, i) => (
                <td key={i} className="border px-4 py-2">
                  {value as ReactNode}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </main>
  )
}
