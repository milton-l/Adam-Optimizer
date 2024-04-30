{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -Wno-name-shadowing #-}
{-# OPTIONS_GHC -Wno-type-defaults #-}
{-# OPTIONS_GHC -Wno-unused-local-binds #-}
{-# OPTIONS_GHC -Wno-unused-matches #-}

module Main where

import Control.Monad.State

newtype Vector a = Vector [a] deriving (Show, Eq)

class VectorOps v where
  vecZero :: Int -> v Double
  vecAdd :: v Double -> v Double -> v Double
  vecSub :: v Double -> v Double -> v Double
  vecMul :: v Double -> v Double -> v Double
  vecDiv :: v Double -> v Double -> v Double
  vecScale :: Double -> v Double -> v Double
  vecMap :: (Double -> Double) -> v Double -> v Double
  vecLength :: v a -> Int

instance VectorOps Vector where
  vecZero n = Vector (replicate n 0)
  vecAdd (Vector xs) (Vector ys) = Vector (zipWith (+) xs ys)
  vecSub (Vector xs) (Vector ys) = Vector (zipWith (-) xs ys)
  vecMul (Vector xs) (Vector ys) = Vector (zipWith (*) xs ys)
  vecDiv (Vector xs) (Vector ys) = Vector (zipWith (/) xs ys)
  vecScale s (Vector xs) = Vector (map (* s) xs)
  vecMap f (Vector xs) = Vector (map f xs)
  vecLength (Vector xs) = length xs

data AdamState = AdamState
  { timestep :: Int,
    m_t :: Vector Double,
    v_t :: Vector Double
  }
  deriving (Show)

data Hyperparameters = Hyperparameters
  { alpha :: Double,
    beta1 :: Double,
    beta2 :: Double,
    epsilon :: Double
  }

type AdamOptimizer = StateT AdamState IO

updateMoments :: Vector Double -> Hyperparameters -> AdamOptimizer ()
updateMoments grads hyperparams = do
  state <- get
  let t = timestep state + 1
      m = m_t state
      v = v_t state
      beta1' = beta1 hyperparams
      beta2' = beta2 hyperparams
  put
    state
      { timestep = t,
        m_t = vecAdd (vecScale beta1' m) (vecScale (1 - beta1') grads),
        v_t = vecAdd (vecScale beta2' v) (vecMap (^ 2) grads)
      }

adamStep :: Vector Double -> Vector Double -> Hyperparameters -> AdamOptimizer (Vector Double)
adamStep params grads hyperparams = do
  updateMoments grads hyperparams
  state <- get
  let alpha' = alpha hyperparams
      epsilon' = epsilon hyperparams
      mhat = vecScale (1 / (1 - beta1 hyperparams ** fromIntegral (timestep state))) (m_t state)
      vhat = vecMap sqrt $ vecScale (1 / (1 - beta2 hyperparams ** fromIntegral (timestep state))) (v_t state)
  return $ vecSub params (vecScale alpha' (vecDiv mhat (vecAdd vhat (vecZero (vecLength params)))))

execAdam :: Vector Double -> Hyperparameters -> Int -> AdamOptimizer (Vector Double)
execAdam params hyperparams 0 = do
  liftIO $ putStrLn ("Final params: " ++ show params)
  return params
execAdam params hyperparams n = do
  liftIO $ putStrLn ("Step " ++ show (n - 1) ++ " updated params: " ++ show params)
  let grads = vecMap (* 2) params
  newParams <- adamStep params grads hyperparams
  execAdam newParams hyperparams (n - 1)

runAdam :: Vector Double -> Hyperparameters -> Int -> AdamState -> IO ()
runAdam params hyperparams n initial =
  void $ evalStateT (execAdam params hyperparams n) initial

main :: IO ()
main = do
  let initialParams = Vector [1.0, 2.0]
      initialState = AdamState 0 (vecZero 2) (vecZero 2)
      hyperparams = Hyperparameters 0.01 0.9 0.999 1.0e-8
  putStrLn "Starting..."
  runAdam initialParams hyperparams 100 initialState
